class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = 48000;
        this.targetSampleRate = 16000;
        this.audioBuffer = [];
        this.bufferDuration = 0.1;
        this.samplesPerBuffer = Math.ceil(this.targetSampleRate * this.bufferDuration);
        this.minAudioLength = 512;
        
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

        // 性能优化
        this.processingEnabled = true;
        this.lastProcessTime = 0;
        
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
        this.minAudioLength = Math.min(512, this.samplesPerBuffer);
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

        const currentTime = performance.now();
        if (currentTime - this.lastProcessTime < 10) {
            return true;
        }
        this.lastProcessTime = currentTime;

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
        if (!float32Data || float32Data.length < this.minAudioLength) {
            return null;
        }

        const downsampledBuffer = this.downsampleBuffer(float32Data, this.sampleRate, this.targetSampleRate);
        
        if (!downsampledBuffer || downsampledBuffer.length === 0) {
            return null;
        }

        const normalizedBuffer = this.normalizeAudio(downsampledBuffer);
        
        const pcmBuffer = this.floatTo16BitPCM(normalizedBuffer);
        
        return pcmBuffer;
    }

    validateAudioData(inputData) {
        if (!inputData || !Array.isArray(inputData)) {
            return null;
        }

        let float32Data;
        if (inputData instanceof Float32Array) {
            float32Data = inputData;
        } else {
            float32Data = new Float32Array(inputData);
        }

        const maxAmplitude = Math.max(...float32Data.map(Math.abs));
        if (maxAmplitude < 0.001) {
            return null;
        }

        return float32Data;
    }

    normalizeAudio(buffer) {
        const maxVal = Math.max(...buffer.map(Math.abs));
        if (maxVal > 0) {
            const factor = 1 / maxVal;
            for (let i = 0; i < buffer.length; i++) {
                buffer[i] *= factor;
            }
        }
        return buffer;
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
        this.audioBuffer.push(...audioData);
        
        while (this.audioBuffer.length >= this.samplesPerBuffer) {
            const chunk = new Int16Array(this.samplesPerBuffer);
            for (let i = 0; i < this.samplesPerBuffer; i++) {
                chunk[i] = this.audioBuffer[i];
            }
            
            this.audioBuffer.splice(0, this.samplesPerBuffer);
            
            this.port.postMessage({
                type: 'audioChunk',
                data: chunk.buffer,
                timestamp: performance.now()
            }, [chunk.buffer]);
        }
    }

    getStatistics() {
        return {
            bufferSize: this.audioBuffer.length,
            speechThreshold: this.vadStates.speechThreshold,
            isSpeechActive: this.vadStates.isSpeechActive,
            sampleRate: this.targetSampleRate
        };
    }
}

registerProcessor('audio-processor', AudioProcessor);
