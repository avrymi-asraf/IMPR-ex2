# Audio Watermarking Exercise

Implementation of audio watermarking techniques for detection and analysis.

## Required Functions

### Core Utils
* [x] Load audio file with correct sampling rate, `load_audio`
* [x] Save audio file with given sampling rate, `save_audio`
* [x] Plot audio signal in the time domain, `visualize_signal`
* [x] Function that calculates the DFT of audio, `DFT`
* [x] Plot audio spectrum in the frequency domain, `visualize_spectrum`
* [x] Function that calculates the STFT of audio, `STFT`
* [x] Function that plots the spectrogram of STFT, `spectrogram_stft`

### Task 1: Watermark Creation
* [ ] Create inaudible but detectable watermark, `create_good_watermark`
* [ ] Create audible watermark example, `create_bad_watermark`
* [ ] Apply watermark to audio signal, `apply_watermark`

### Task 2: Watermark Classification
* [ ] Extract watermark from audio signal, `detect_watermark`
* [ ] Group audio files by watermark type, `classify_watermarks`
* [ ] Identify watermark creation function, `extract_watermark_function`

### Task 3: Audio Speedup Analysis
* [ ] Check frequency domain modifications, `detect_frequency_modifications`
* [ ] Check time domain modifications, `detect_time_modifications`
* [ ] Determine speedup method and ratio, `analyze_speedup_method`


## Detailed Task Description

### Task 1: Creating Watermarks
**Goal**: Learn how to embed hidden information in audio while maintaining quality
- Create two types of watermarks to understand the trade-off between detectability and perceptibility
- Demonstrate how incorrect watermarking can degrade audio quality
- Practice frequency-domain signal manipulation

### Task 2: Watermark Classification
**Goal**: Develop signal analysis skills and pattern recognition abilities
- Learn to extract hidden patterns from complex signals
- Practice grouping and classification based on signal properties
- Understand how different watermarking functions affect the signal
- Gain experience in reverse engineering signal modifications

### Task 3: Audio Speedup Analysis
**Goal**: Understand the differences between time and frequency domain operations
- Learn to identify different types of audio modifications
- Practice analyzing signal characteristics in both domains
- Understand how signal properties change under different transformations
- Develop empirical analysis techniques for audio processing

### Implementation Ideas

#### Watermark Creation
- Use frequency domain for "good" watermark (high frequencies are less audible)
- Consider amplitude modulation for watermark strength
- Look into psychoacoustic masking principles
- Test different frequency bands for optimal imperceptibility

#### Watermark Detection
- Compare frequency spectrums before/after watermarking
- Look for consistent patterns across same-watermark files
- Consider using signal correlation for detection
- Statistical analysis of signal properties

#### Speedup Analysis
- Analyze harmonic relationships in the frequency domain
- Check signal envelope preservation
- Compare energy distribution across frequencies
- Look for artifacts specific to each speedup method