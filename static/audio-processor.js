/**
 * AudioWorklet 运行在 AudioWorkletGlobalScope，与 window 不同：
 * 规范/实现里长期不保证存在全局 performance，直接写 performance.now 会 ReferenceError。
 * 这里只用 Date.now（~1ms 精度）；Worklet 内请勿使用裸的 performance。
 */
function workletNowMs() {
    return Date.now();
}

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 48000;
        this.targetSampleRate = 16000;
        this.audioBuffer = [];
        this.bufferDuration = 0.1;
        this.samplesPerBuffer = Math.ceil(this.targetSampleRate * this.bufferDuration);
        /** @type {Int16Array|null} */
        this._acc = null;
        this._accLen = 0;
        
        // VAD状态管理
        this.vadStates = {
            speechProbabilities: [],
            speechThreshold: 0.3,
            minSpeechDuration: 0.1,
            maxSilenceDuration: 0.5,
            currentSpeechStart: null,
            isSpeechActive: false,
            silenceCounter: 0
        };

        this.processingEnabled = true;
        
        this.port.onmessage = (event) => {
            if (event.data.type === 'init') {
                this.sampleRate = event.data.sampleRate || 48000;
                this.targetSampleRate = event.data.targetSampleRate || 16000;
                this.recalculateBufferParams();
                console.log('AudioWorklet初始化完成，采样率:', this.sampleRate, '目标采样率:', this.targetSampleRate);
            } else if (event.data.type === 'setThreshold') {
                this.vadStates.speechThreshold = event.data.threshold || 0.3;
                console.log('VAD阈值更新:', this.vadStates.speechThreshold);
            } else if (event.data.type === 'reset') {
                this.resetState();
                console.log('AudioWorklet状态已重置');
            }
        };
    }

    recalculateBufferParams() {
        this.samplesPerBuffer = Math.ceil(this.targetSampleRate * this.bufferDuration);
    }

    resetState() {
        this.audioBuffer = [];
        this.vadStates.speechProbabilities = [];
        this.vadStates.currentSpeechStart = null;
        this.vadStates.isSpeechActive = false;
        this.vadStates.silenceCounter = 0;
    }

    process(inputs, outputs, parameters) {
        if (!this.processingEnabled) {
            return true;
        }

        const input = inputs[0];
        if (input && input.length > 0) {
            const inputData = input[0];
            if (!inputData || inputData.length === 0) {
                return true;
            }

            const processedAudio = this.processAudioChunk(inputData);
            
            if (processedAudio && processedAudio.length > 0) {
                this.accumulateAndSendAudio(processedAudio);
            }
        }

        return true;
    }

    processAudioChunk(inputData) {
        const float32Data = this.validateAudioData(inputData);
        if (!float32Data || float32Data.length === 0) {
            return null;
        }

        const downsampledBuffer = this.downsampleBuffer(float32Data, this.sampleRate, this.targetSampleRate);
        
        if (!downsampledBuffer || downsampledBuffer.length === 0) {
            return null;
        }

        // 勿对每一渲染块做峰值归一化，会扭曲短时包络，极易把「你好」等读成单字错识
        const pcmBuffer = this.floatTo16BitPCM(downsampledBuffer);
        
        return pcmBuffer;
    }

    validateAudioData(inputData) {
        if (!inputData) {
            return null;
        }

        let float32Data;
        if (inputData instanceof Float32Array) {
            float32Data = inputData;
        } else if (Array.isArray(inputData)) {
            float32Data = new Float32Array(inputData);
        } else {
            return null;
        }

        let maxAmplitude = 0;
        for (let i = 0; i < float32Data.length; i++) {
            const a = float32Data[i];
            const abs = a < 0 ? -a : a;
            if (abs > maxAmplitude) maxAmplitude = abs;
        }
        if (maxAmplitude < 0.001) {
            return null;
        }

        return float32Data;
    }

    downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
        if (outputSampleRate >= inputSampleRate) {
            return buffer;
        }

        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        
        if (newLength === 0) {
            return new Float32Array(0);
        }

        const result = new Float32Array(newLength);
        
        if (sampleRateRatio === 2) {
            this.downsampleBy2(buffer, result);
        } else {
            this.downsampleByRatio(buffer, result, sampleRateRatio);
        }
        
        return result;
    }

    downsampleBy2(input, output) {
        const len = output.length;
        for (let i = 0; i < len; i++) {
            const j = i * 2;
            output[i] = (input[j] + (input[j + 1] || input[j])) * 0.5;
        }
    }

    downsampleByRatio(buffer, result, ratio) {
        const len = result.length;
        for (let offsetResult = 0; offsetResult < len; offsetResult++) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
            let accum = 0, count = 0;
            
            for (let i = offsetResult === 0 ? 0 : Math.round(offsetResult * ratio); 
                 i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            
            result[offsetResult] = count > 0 ? accum / count : 0;
        }
    }

    floatTo16BitPCM(input) {
        const len = input.length;
        const output = new Int16Array(len);
        
        for (let i = 0; i < len; i++) {
            let s = Math.max(-1, Math.min(1, input[i]));
            if (s < 0) {
                output[i] = Math.round(s * 0x8000);
            } else {
                output[i] = Math.round(s * 0x7FFF);
            }
        }
        
        return output;
    }

    accumulateAndSendAudio(audioData) {
        const m = this.samplesPerBuffer;
        const addLen = audioData.length;
        let acc = this._acc;
        let accLen = this._accLen;
        const need = accLen + addLen;
        if (!acc || acc.length < need) {
            const grow = Math.max(need, Math.max(m * 4, acc ? acc.length * 2 : 0));
            const next = new Int16Array(grow);
            if (accLen > 0) next.set(acc.subarray(0, accLen), 0);
            acc = next;
        }
        acc.set(audioData, accLen);
        accLen += addLen;
        this._acc = acc;
        this._accLen = accLen;

        let consumed = 0;
        while (accLen - consumed >= m) {
            const chunk = new Int16Array(m);
            chunk.set(acc.subarray(consumed, consumed + m));
            consumed += m;
            this.port.postMessage({
                type: 'audioChunk',
                data: chunk.buffer,
                timestamp: workletNowMs()
            }, [chunk.buffer]);
        }
        if (consumed > 0) {
            const remain = accLen - consumed;
            if (remain > 0) {
                acc.copyWithin(0, consumed, accLen);
            }
            this._accLen = remain;
        }
    }

    getStatistics() {
        return {
            bufferSize: this._accLen,
            speechThreshold: this.vadStates.speechThreshold,
            isSpeechActive: this.vadStates.isSpeechActive,
            sampleRate: this.targetSampleRate
        };
    }
}

registerProcessor('audio-processor', AudioProcessor);
