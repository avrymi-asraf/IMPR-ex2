import numpy as np
from Tools import load_audio, STFT, plot_spectrogram
from typing import Tuple, Optional
import plotly.express as px
import os


def extract_frequency_range(
    stft_magnitudes: np.ndarray, frequencies: np.ndarray, frequency_range: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter the STFT magnitudes to retain only the specified frequency range.

    Args:
        stft_magnitudes (np.ndarray): 2D array of STFT magnitudes with rows as frequency bins and columns as time points.
        frequencies (np.ndarray): 1D array of frequency bins for the STFT.
        freq_range (list[float]): List containing the lower and upper bounds of the frequency range to retain [low_freq, high_freq].

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered STFT magnitudes and the corresponding frequency range.
    """
    low_freq, high_freq = frequency_range
    freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    filtered_magnitudes = stft_magnitudes[freq_mask, :]
    filtered_frequencies = frequencies[freq_mask]
    return filtered_magnitudes, filtered_frequencies


def waveform_from_frequency_range(
    audio_data: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    frequency_range: list,
) -> np.ndarray:
    """
    Extract a range of frequencies from the audio signal and convert it back to a waveform.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        window_size (int): Size of the FFT window
        hop_size (int): Hop size between successive windows
        frequency_range (list): List containing the lower and upper bounds of the frequency range to extract [low_freq, high_freq]

    Returns:
        np.ndarray: Extracted waveform containing only the specified frequency range
    """
    times, frequencies, stft_magnitudes = STFT(
        audio_data, sample_rate, window_size, hop_size
    )
    filtered_magnitudes, filtered_frequencies = extract_frequency_range(
        stft_magnitudes, frequencies, frequency_range
    )
    extracted_wave = (filtered_magnitudes.T * filtered_frequencies).T.sum(axis=0)
    return extracted_wave


def dft_on_frequency_range(
    audio_data: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    frequency_range: list,
) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) of the extracted waveform within a specified frequency range.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        window_size (int): Size of the FFT window
        hop_size (int): Hop size between successive windows
        freq_range (list): List containing the lower and upper bounds of the frequency range to extract [low_freq, high_freq]

    Returns:
        np.ndarray: DFT of the extracted waveform within the specified frequency range
    """
    extracted_wave = waveform_from_frequency_range(
        audio_data, sample_rate, window_size, hop_size, frequency_range
    )
    dft_result = np.fft.fft(extracted_wave)
    return dft_result


def process_frequency_range(
    file_path: str,
    window_size: int,
    extract_frequency: Tuple[int, int],
    show_graphs: Tuple[bool, bool, bool],
) -> None:
    """
    Process a single audio file and plot the time-domain waveform,
    frequency-domain spectrum, and filtered spectrogram based on the show_graphs argument.

    Args:
        file_path (str): Path to the audio file.
        window_size (int): Size of the window for STFT.
        extract_frequency (Tuple[int, int]): Frequency range to extract.
        show_graphs (Tuple[bool, bool, bool]): Tuple indicating which graphs to show (spectrogram,time-domain, frequency-domain).

    Returns:
        wave_spectrum (np.ndarray): Frequency-domain spectrum of the extracted waveform.
    """
    audio_data, sample_rate = load_audio(file_path)
    hop_size = window_size // 2

    # Compute STFT
    times, frequencies, magnitudes = STFT(
        audio_data, sample_rate, window_size, hop_size
    )
    filtered_magnitudes, filtered_frequencies = extract_frequency_range(
        magnitudes, frequencies, extract_frequency
    )

    # Compute time-domain waveform
    wave = (filtered_magnitudes.T * filtered_frequencies).T.sum(axis=0)
    wave = wave / np.max(wave) - 0.5
    wave_spectrum = np.abs(np.fft.fft(wave))

    # Plot graphs based on show_graphs argument
    if show_graphs[0]:
        plot_spectrogram(
            times,
            filtered_frequencies,
            filtered_magnitudes,
            title=f"Filtered Spectrogram of {os.path.basename(file_path)}",
            log_scale=True,
        )
    if show_graphs[1]:
        px.line(
            x=times,
            y=wave,
            title=f"Time-Domain Waveform of {os.path.basename(file_path)}",
        ).show()
    if show_graphs[2]:
        px.bar(
            x=times,
            y=np.abs(wave_spectrum),
            title=f"Frequency-Domain Spectrum of {os.path.basename(file_path)}",
        ).show()
    return wave_spectrum


def detect_watermark_frequency(audio_path: str) -> float:
    """
    Detect and calculate the frequency of a watermark in an audio file, normalized by audio duration.
    
    Args:
        audio_path (str): Path to the audio file containing the watermark.
        
    Returns:
        float: Normalized watermark frequency value, calculated by dividing the peak 
              frequency index (in range 5-20) by the audio duration in seconds.
    """
    # Get audio duration
    audio_data, sample_rate = load_audio(audio_path)
    duration = len(audio_data) / sample_rate
    
    # Process frequency range
    high_wave_magnitude = process_frequency_range(
        audio_path, 2**15, (18_000, 21_000), (False, False, False)
    )
    watermark_frequency = np.argmax(high_wave_magnitude[5:20])+5
    
    return watermark_frequency/duration


def mean_watermark_frequency(file_paths: list[str]) -> float:
    """
    Calculate the mean watermark frequency across multiple audio files.
    
    Args:
        file_paths (list[str]): List of paths to audio files to process
        
    Returns:
        float: Mean watermark frequency across all provided files
        
    Raises:
        ValueError: If file_paths is empty
    """
    if not file_paths:
        raise ValueError("File paths list cannot be empty")
        
    frequencies = [detect_watermark_frequency(file_path) for file_path in file_paths]
    return np.mean(frequencies)

