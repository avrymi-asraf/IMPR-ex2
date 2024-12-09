import numpy as np

def add_wave(audio_signal: np.ndarray, frequency: float, amplitude: float, sample_rate: int, 
             indices: list) -> np.ndarray:
    """
    Add a sinusoidal wave to an audio signal between specified index pairs.
    
    Args:
        audio_signal (np.ndarray): Input audio signal
        frequency (float): Frequency of the wave to add in Hz
        amplitude (float): Amplitude/intensity of the wave
        sample_rate (int): Sampling rate of the audio in Hz
        indices (list): List of indices to add wave between odd indices and the following number
        
    Returns:
        np.ndarray: Audio signal with added wave in specified ranges
        
    Raises:
        ValueError: If indices are invalid
    """
    # Create time array for full signal
    duration = len(audio_signal) / sample_rate
    t = np.linspace(0, duration, len(audio_signal))
    
    # Initialize wave array with zeros
    wave = np.zeros_like(audio_signal)
    
    # Validate indices
    if any(idx < 0 or idx >= len(audio_signal) for idx in indices):
        raise ValueError("Invalid indices specified")
    
    # Add wave between odd indices and the following number
    for i in range(1, len(indices), 2):
        start_idx = indices[i-1]
        end_idx = indices[i]
        if start_idx >= end_idx:
            raise ValueError("Invalid index range specified")
        
        # Generate sinusoidal wave only for specified range
        t_range = t[start_idx:end_idx]
        wave[start_idx:end_idx] = amplitude * np.sin(2 * np.pi * frequency * t_range)
    
    # Add wave to original signal
    watermarked_signal = audio_signal + wave
    
    # Normalize to prevent clipping
    watermarked_signal = watermarked_signal / np.max(np.abs(watermarked_signal))
    
    return watermarked_signal


