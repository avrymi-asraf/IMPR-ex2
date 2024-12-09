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
* [x] Plot spectrogram of audio signal, `plot_spectrogram`

### Task 1: Watermark Creation
* [x] Adding a wave to an array that represents audio should be possible, with the range of the addition being its frequency and intensity. , `add_wave`


### Task 2: Watermark Classification
* [x] Extract a range of frequencies from the audio signal and convert it back to a waveform, `waveform_from_frequency_range`
* [x] Compute the Discrete Fourier Transform (DFT) of the extracted waveform within a specified frequency range, `dft_on_frequency_range`
* [x] Filter the STFT magnitudes to retain only the specified frequency range, `extract_frequency_range`
* [x] Classify an audio file based on the frequency of its watermark, `classify_by_frequency_watermark`


### Task 3: Audio Speedup Analysis
* [ ] Check frequency domain modifications, `detect_frequency_modifications`
* [ ] Check time domain modifications, `detect_time_modifications`
* [ ] Determine speedup method and ratio, `analyze_speedup_method`


## Detailed Task Description

### Task 1: Creating Watermarks
**Goal**:add watermark to audio, create good watermark and wrong watermark.
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
