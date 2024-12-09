import numpy as np
import scipy.io.wavfile as wavfile
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Optional
import soundfile as sf


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return signal data with sampling rate.

    Args:
        file_path (str): Path to the audio file (supports various formats including .wav, .flac, .ogg)

    Returns:
        Tuple[np.ndarray, int]: Audio signal array (normalized float) and sampling rate

    Raises:
        ValueError: If file cannot be read or format is not supported
    """
    try:
        audio_data, sample_rate = sf.read(file_path)
        audio_data = audio_data.astype(np.float32)
        return audio_data, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")


def save_audio(file_path: str, audio_data: np.ndarray, sample_rate: int) -> None:
    """
    Save audio data to WAV file.

    Args:
        file_path (str): Output file path (.wav format)
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz

    Raises:
        ValueError: If data cannot be written to file
    """
    try:
        audio_data = (audio_data * 32768.0).astype(np.int16)
        wavfile.write(file_path, sample_rate, audio_data)
    except Exception as e:
        raise ValueError(f"Error saving audio file: {str(e)}")


def visualize_signal(
    audio_data: np.ndarray, sample_rate: int, title: Optional[str] = "Audio Signal"
) -> None:
    """
    Plot audio signal in time domain using Plotly.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        title (str, optional): Plot title
    """
    time = np.arange(len(audio_data)) / sample_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode="lines", name="Audio Signal"))
    fig.update_layout(
        title=title, xaxis_title="Time (seconds)", yaxis_title="Amplitude"
    )
    fig.show()


def DFT(audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency spectrum of the entire audio signal.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz

    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequencies and corresponding spectrum magnitudes
    """
    frequencies = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(audio_data))
    return frequencies, spectrum


def visualize_spectrum(
    audio_data: np.ndarray,
    sample_rate: int,
    title: Optional[str] = "Frequency Spectrum",
) -> None:
    """
    Plot audio spectrum in frequency domain using Plotly.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        title (str, optional): Plot title
    """
    frequencies, spectrum = DFT(audio_data, sample_rate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies, y=spectrum, mode="lines", name="Spectrum"))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        yaxis_type="log",
    )
    fig.show()


def STFT(
    audio_data: np.ndarray, sample_rate: int, window_size: int, hop_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) of the audio signal.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        window_size (int): Size of the FFT window
        hop_size (int): Hop size between successive windows

    Returns:
        - Times (np.ndarray): 1D array of time points at which STFT is computed.
        - Frequencies (np.ndarray): 1D array of frequency bins for the STFT.
        - STFT Magnitudes (np.ndarray): 2D array of STFT magnitudes with rows as frequency bins and columns as time points.
    """
    from scipy.signal import stft

    frequencies, times, Zxx = stft(
        audio_data, fs=sample_rate, nperseg=window_size, noverlap=window_size - hop_size
    )
    stft_magnitudes = np.abs(Zxx)
    return times, frequencies, stft_magnitudes


def plot_spectrogram(
    times: np.ndarray,
    frequencies: np.ndarray,
    stft_magnitudes: np.ndarray,
    title: Optional[str] = "Spectrogram",
    log_scale: bool = False,
) -> None:
    """
    Plot the spectrogram from STFT output using Plotly.

    Args:
        times (np.ndarray): 1D array of time points at which STFT is computed.
        frequencies (np.ndarray): 1D array of frequency bins for the STFT.
        stft_magnitudes (np.ndarray): 2D array of STFT magnitudes with rows as frequency bins and columns as time points.
        title (str, optional): Plot title
        log_scale (bool, optional): Whether to display the frequency axis in log scale
    """
    fig = px.imshow(
        20 * np.log10(stft_magnitudes),
        x=times,
        y=frequencies,
        origin="lower",
        aspect="auto",
        color_continuous_scale="viridis",
    )
    fig.update_layout(
        title=title, xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)"
    )
    if log_scale:
        fig.update_yaxes(type="log")
    fig.show()


def spectrogram_stft(
    audio_data: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    title: Optional[str] = "Spectrogram (STFT)",
    log_scale: bool = False,
) -> None:
    """
    Plot the spectrogram of the audio signal using STFT and Plotly.

    Args:
        audio_data (np.ndarray): Audio signal array
        sample_rate (int): Sampling rate in Hz
        window_size (int): Size of the FFT window
        hop_size (int): Hop size between successive windows
        title (str, optional): Plot title
        log_scale (bool, optional): Whether to display the frequency axis in log scale
    """
    times, frequencies, stft_magnitudes = STFT(
        audio_data, sample_rate, window_size, hop_size
    )
    plot_spectrogram(times, frequencies, stft_magnitudes, title, log_scale)


if __name__ == "__main__":
    # Test the functions with a sample audio file
    test_file_path = "data\\Task 1\\task1.wav"
    output_file_path = "output_audio.wav"

    # Load audio
    audio_data, sample_rate = load_audio(test_file_path)

    # Save audio
    save_audio(output_file_path, audio_data, sample_rate)

    # Visualize signal
    visualize_signal(audio_data, sample_rate, title="Test Audio Signal")

    # Visualize spectrum
    visualize_spectrum(audio_data, sample_rate, title="Test Frequency Spectrum")

    # Compute and visualize spectrogram
    window_size = 1024
    hop_size = 512
    spectrogram_stft(
        audio_data,
        sample_rate,
        window_size,
        hop_size,
        title="Test Spectrogram (STFT)",
        log_scale=True,
    )


