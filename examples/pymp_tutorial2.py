"""

Tutorial provided as part of PyMP

M. Moussallam

"""
from PyMP.mdct import Dico, LODico
from PyMP.mdct.rand import SequenceDico
from PyMP import mp, mp_coder, Signal
signal = Signal('../data/ClocheB.wav', mono=True)  # Load Signal
signal.crop(0, 4.0 * signal.fs)     # Keep only 4 seconds
# atom of scales 8, 64 and 512 ms
scales = [(s * signal.fs / 1000) for s in (8, 64, 512)]
signal.pad(scales[-1])
# Dictionary for Standard MP
dico = Dico(scales)
# Launching decomposition, stops either at 20 dB of SRR or 2000 iterations
app, dec = mp.mp(signal, dico, 20, 2000, pad=False)

app.atom_number

snr, bitrate, quantized_app = mp_coder.simple_mdct_encoding(
    app, 8000, Q=14)
print (snr, bitrate)

print "With Q=5"
snr, bitrate, quantized_app = mp_coder.simple_mdct_encoding(
    app, 8000, Q=5)
print (snr, bitrate)


snr, bitrate, quantized_app = mp_coder.simple_mdct_encoding(
    app, 2000, Q=14)
print (snr, bitrate)

lomp_dico = LODico(scales)
lomp_app, lomp_dec = mp.mp(
    signal, lomp_dico, 20, 2000, pad=False)
lomp_snr, lomp_bitrate, lomp_quantized_app = mp_coder.simple_mdct_encoding(
    lomp_app, 2000, Q=14, shift_penalty=True)
print (lomp_snr, lomp_bitrate)

rssmp_dico = SequenceDico(scales)
rsssmp_app, rssmp_dec = mp.mp(signal, rssmp_dico, 20, 2000, pad=False)
rssmp_snr, rssmp_bitrate, rssmp_quantized_app = mp_coder.simple_mdct_encoding(
    rsssmp_app, 2000, Q=14)
print (rssmp_snr, rssmp_bitrate)

print (quantized_app.atom_number, lomp_quantized_app.atom_number,
       rssmp_quantized_app.atom_number)

# quantized_app.plotTF()
# plt.show()


print " now at a much larger level : a snr of nearly 40 dB and around 20 Kbps"
app, dec = mp.mp(signal, dico, 40, 20000, pad=False)
snr, bitrate, quantized_app = mp_coder.simple_mdct_encoding(
    app, 64000, Q=16)
print (snr, bitrate)

del app, dico

lomp_app, lomp_dec = mp.mp(
    signal, lomp_dico, 40, 20000, pad=False)
lomp_snr, lomp_bitrate, lomp_quantized_app = mp_coder.simple_mdct_encoding(
    lomp_app, 64000, Q=16, shift_penalty=True)
print (lomp_snr, lomp_bitrate)

del lomp_app, lomp_dico

rsssmp_app, rssmp_dec = mp.mp(
    signal, rssmp_dico, 40, 20000, pad=False)
rssmp_snr, rssmp_bitrate, rssmp_quantized_app = mp_coder.simple_mdct_encoding(
    rsssmp_app, 64000, Q=16)
print (rssmp_snr, rssmp_bitrate)
