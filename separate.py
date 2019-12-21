import librosa as lb
import numpy as np

def separate(audioIn, k_max):
    """
    Function to separate harmonic and percussive components from audio signal.
    Audio signal is defined by 'audioIn'. Max number of iterations is defined by 'k_max'.
    Returns separated signals and spectrograms for original, harmonic and percussive signals.
    """

    gamma = 0.3
    alpha = 0.5

    # Compute stft
    F = lb.core.stft(audioIn, n_fft=1024, hop_length=512)

    # Range-compressed version of the power spectrogram
    W = abs(F)**(2*gamma)

    # Initial values for harmonic (H) and percussive (P) components
    H = 0.5*W
    P = 0.5*W

    # Start updating H and P
    l = H.shape[0]
    w = H.shape[1]
    for k in range(k_max):
        delta = []
        H_new = np.zeros((l,w))
        P_new = np.zeros((l,w))

        # Calculate update variables
        for i in range(1, W.shape[1]-1):
            frame = []
            for h in range(1, W.shape[0]-1):
                d = alpha*((H[h][i-1] - 2*H[h][i] + H[h][i+1])/4) - (1-alpha)*((P[h-1][i] - 2*P[h][i] + P[h+1][i])/4)
                frame.append(d)
            delta.append(frame)

        # Calculate updated H and P
        for i in range(1, W.shape[1]-1):
            for h in range(1, W.shape[0]-1):
                H_new[h][i] = min(max(H[h][i] + delta[i-1][h-1], 0), W[h][i])
                P_new[h][i] = W[h][i] - H_new[h][i]

        # Save previous values for binarization
        H_old = H 
        P_old = P
        
        # Update H and P
        H = H_new
        P = P_new

    # Binarize results
    H_bin = np.zeros((l,w))
    P_bin = np.zeros((l,w))
    for i in range(1, W.shape[1]-1):
        for h in range(1, W.shape[0]-1):
            if H_old[h][i] < P_old[h][i]:
                H_bin[h][i] = 0
                P_bin[h][i] = W[h][i]
            else:
                H_bin[h][i] = W[h][i]
                P_bin[h][i] = 0

    # Convert to waveforms
    H_wav = np.zeros((l, w), dtype=complex)
    P_wav = np.zeros((l, w), dtype=complex)
    for i in range(1, W.shape[1]-1):
        for h in range(1, W.shape[0]-1):
            H_wav[h][i] = H_bin[h][i]**(1/(2*gamma))*F[h][i]/abs(F[h][i])
            P_wav[h][i] = P_bin[h][i]**(1/(2*gamma))*F[h][i]/abs(F[h][i])
            if np.isnan(H_wav[h][i].real):
                H_wav[h][i] = 0
            if np.isnan(P_wav[h][i].real):
                P_wav[h][i] = 0

    # Inverse stft
    H_result = lb.core.istft(H_wav, hop_length=512)
    P_result = lb.core.istft(P_wav, hop_length=512)

    # Evaluation
    s = audioIn[:len(H_result)] #original
    e_h = s - H_result #original minus harmonic
    e_p = s - P_result #original minus percussive
    
    # Calculate SNRs
    snr_h = 10*np.log10((sum(s**2)/(sum(e_h**2))))
    snr_p = 10*np.log10((sum(s**2)/(sum(e_p**2))))
    print("Harmonic SNR: %.1f, Percussive SNR: %.1f" %(snr_h,snr_p))
    
    # Return separation results and spectrograms
    return H_result, P_result, [W, H_bin, P_bin]