# CompE Project Lab 3 Group 8 Project
# Code by Mason Marnell, Chase Ohlenburger, and Giulia Piombo
# Inspired by this blog post by Fraida Fund: https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/

from rtlsdr import *
from pylab import *
from matplotlib.animation import FuncAnimation
import time
import threading
import pyaudio
import struct
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import PySimpleGUI as sg

import matplotlib
from sympy import true

sdr = RtlSdr()
p = pyaudio.PyAudio()

def decoder():
    global samples, F_offset, Fs, f_bw, N, FreqInput, stream, bits, lenErrorFix
    # Commercial FM and Weather Radio
    if(BandSelect == 1 or BandSelect == 2):
        matplotlib.use('Agg')  # necessary for headless mode
        F_station = int(FreqInput*1000000)
        F_offset = 250000
        Fc = F_station - F_offset           # Center Freq
        Fs = int(1140000)                   # Sample Rate
        N = int(1024000)                    # Sample Number
        # Commercial FM and weather has a WORKING bandwidth of 200 kHz
        f_bw = 200000

        stream = p.open(format=pyaudio.paInt16, channels=1,
                        rate=45120, output=True)

    # FRS Radio
    elif(BandSelect == 3):
        matplotlib.use('Agg')  # necessary for headless mode
        # Center Freq, FRS has no offset     462.61250 is channel 3
        Fc = int(FreqInput*1000000)
        Fs = int(1140000)                   # Sample Rate
        N = int(1024000)                    # Sample Number
        f_bw = 25000                        # FRS has a working bandwidth of 25 kHz, even though theoretically it is less

        audio_rate = f_bw*0.99
        stream = p.open(format=pyaudio.paInt16, channels=1,
                        rate=int(audio_rate), output=True)

    # Graph Plot, not used in the current implementation
    elif(BandSelect == 4):
        sample_rate = 2.56e6  # Hz,
        center_freq = 462.6e6  # Hz
        fft_size = 512  # power of 2, decreasing this increases performance
        num_samps = 256*fft_size

        # sdr = RtlSdr()
        sdr.sample_rate = int(sample_rate)
        sdr.center_freq = int(center_freq)

        samples = sdr.read_samples(num_samps)

        fig = plt.figure()

        def updatefig(i):
            samples = sdr.read_samples(num_samps)

            fig.clear()

            psd(samples, NFFT=fft_size, Fs=sdr.sample_rate /
                1e6, Fc=sdr.center_freq/1e6)

            xlabel('Frequency (MHz)')
            ylabel('Relative power (dB)')

        ani = FuncAnimation(fig, updatefig, interval=1000, blit=False)
        plt.show()


    # configure device
    sdr.sample_rate = Fs      # Hz
    sdr.center_freq = Fc      # Hz
    sdr.gain = 'auto'

    samples = np.array(0)
    bits = bytes([])

    lenErrorFix = 0 #chase

    def takeSamples():
        global samples
        print("samples start")
        samples = sdr.read_samples(N)
        print("samples done")

    def sampleMathFM():  # could also use this to update a plot for the FFT
        print("math start")
        # Convert samples to a numpy array
        x1 = np.array(samples).astype("complex64")

        # To mix the data down, generate a digital complex exponential
        # (with the same length as x1) with phase -F_offset/Fs
        if(lenErrorFix>=1): #chase
            fc1 = np.exp(-1.0j*2.0*np.pi* F_offset/Fs*np.arange(len(x1)))
            # Now, just multiply x1 and the digital complex expontential  
            x2 = x1 * fc1

            dec_rate = int(Fs / f_bw)
            x4 = signal.decimate(x2, dec_rate)  
            Fs_y = Fs/dec_rate #new sampling rate


            ### Polar discriminator - turns the complex array into real values using a phase correlator
            y5 = x4[1:] * np.conj(x4[:-1])  
            x5 = np.angle(y5)

            # The de-emphasis filter
            # Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
            d = Fs_y * 75e-6   # Calculate the # of samples to hit the -3dB point 
            # In America, a 75us time constant is used for both mono and stereo 
            x = np.exp(-1/d)   # Calculate the decay between each sample  
            b = [1-x]          # Create the filter coefficients  
            a = [1,-x]  
            x6 = signal.lfilter(b,a,x5)  

            # Find a decimation rate to achieve audio sampling rate between 44-48 kHz
            audio_freq = 44100.0  
            dec_audio = int(Fs_y/audio_freq)  
            Fs_audio = Fs_y / dec_audio

            x7 = signal.decimate(x6, dec_audio)  

            # Scale audio to adjust volume
            x7 *= 10000 / np.max(np.abs(x7))  
            # Save to file as 16-bit signed single-channel audio samples
            
            x7 = x7.astype("int16") 
            global bits
            bits = struct.pack(('<%dh' % len(x7)), *x7)

            print(len(bits))
            #time.sleep(0.1)
            print("math done")

    def sampleMathFRS():  # could also use this to update a plot for the FFT
        print("math start")
        # Convert samples to a numpy array
        x1 = np.array(samples).astype("complex64")

        # To mix the data down, generate a digital complex exponential
        # (with the same length as x1) with phase -F_offset/Fs
        if(lenErrorFix>=1): #chase   
            fc1 = np.exp(-1.0j*2.0*np.pi* 1/Fs*np.arange(len(x1)))  
            # Now, just multiply x1 and the digital complex expontential
            x2 = x1 * fc1

            #f_bw = 25000 #FM has a bandwidth of 200 kHz #### moved this up there^
            dec_rate = int(Fs / f_bw)
            x4 = signal.decimate(x2, dec_rate)  
            Fs_y = Fs/dec_rate #new sampling rate
            



            ### Polar discriminator - turns the complex array into real values using a phase correlator
            y5 = x4[1:] * np.conj(x4[:-1])  
            x5 = np.angle(y5)

            # No de-emphasis filter needed for FRS

            # Results also seem better with no filter. Maybe research more 

            x7 = x5  #skip decimation for FRS

            # Scale audio to adjust volume
            x7 *= 16000 / np.max(np.abs(x7))  
            # Save to file as 16-bit signed single-channel audio samples
            x7 = x7.astype("int16") 
            print("len of x7: ")
            print(len(x7))

            global bits
            bits = struct.pack(('<%dh' % len(x7)), *x7)
            print(len(bits))
            #time.sleep(0.1)
            print("math done")

    def playAudio():
        global stream
        global bits
        print("audio start")
        bitsAudio = bits
        stream.write(bitsAudio)
        print("audio done")

    stop = 0
    if(BandSelect == 1 or BandSelect == 2):
        while stop <= 14:
            print("main loop 1 start")
            t1 = threading.Thread(target=takeSamples)
            t2 = threading.Thread(target=sampleMathFM)
            t3 = threading.Thread(target=playAudio)
            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()
            lenErrorFix=1 #chase
            stop += 1
            print("main loop 1 done")

    elif(BandSelect == 3):
        while stop <= 14:
            print("main loop 2 start")
            t1 = threading.Thread(target=takeSamples)
            t2 = threading.Thread(target=sampleMathFRS)
            t3 = threading.Thread(target=playAudio)
            t1.start()
            t2.start()
            t3.start()
            t1.join()
            t2.join()
            t3.join()
            lenErrorFix=1 #chase
            stop += 1
            print("main loop 2 done")

def FM_window():
    global BandSelect, FreqInput
    BandSelect = 1
    sg.theme('DarkGrey6')
    button_size = (13,3)
    sg.set_options(font = 'Franklin 20')
    layout = [
        [sg.Text("Pick a FM Radio Station")],
        [sg.Button("93.1", size = button_size), sg.Button("94.5", size = button_size),
            sg.Button("96.3", size = button_size), sg.Button("98.7", size = button_size)],
        [sg.Button("100.7", size = button_size), sg.Button("102.5", size = button_size), 
            sg.Button("106.5", size = button_size), sg.Button("107.7", size = button_size)],
        [sg.Text("")],
        [sg.Text("Enter station"), sg.Input(key = "-INPUT-")],
        [sg.Button("Submit", size = (20,1), key = '-SUBMIT-')],
        [sg.Text("")],
        [sg.Text("Now playing:"), sg.Text(key = '-TEXT-')],
        [sg.Button("Back", size =(20,1))]
    ]

    window = sg.Window("FM Radio Decoder", layout, size=(1000,600), margins=(20,10), element_justification="c")

    current_num = []

    while True:
        event, values = window.read()

        if event == "-SUBMIT-":
            current_num = []
            current_num.append(values['-INPUT-'])
            num_string = ''.join(current_num)
            FreqInput = float(num_string)
            window['-TEXT-'].update(num_string)
            decoder()

        if event == "93.1":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 93.1
            decoder()
        
        if event == "94.5":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 94.5
            decoder()
        
        if event == "96.3":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 96.3
            decoder()

        if event == "98.7":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 98.7
            decoder()

        if event == "100.7":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 100.7
            decoder()

        if event == "102.5":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 102.5
            decoder()

        if event == "106.5":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 106.5
            decoder()

        if event == "107.7":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 107.7
            decoder()

        if event == "Back" or event == sg.WIN_CLOSED:
            break

    window.close()

def Weather_Radio():
    global BandSelect, FreqInput
    BandSelect = 2
    sg.theme('DarkGrey6')
    sg.set_options(font = 'Franklin 20')
    layout = [
        [sg.Text("Pick a Weather Station (depends on your location)")],
        [sg.Text("Lubbock/Dallas/Austin: 162.400")],
        [sg.Text("San Antonio/Amarillo: 162.550")],
        [sg.Text("")],
        [sg.Text("Enter frequency"), sg.Input(key = "-INPUT-")],
        [sg.Button("Submit", size = (20,1), key = '-SUBMIT-')],
        [sg.Text("")],
        [sg.Text("Now playing:"), sg.Text(key = '-TEXT-')],
        [sg.Button("Back", size = (20,1))]
    ]

    window = sg.Window("Weather Station Decoder", layout, size=(1000,600), margins=(20,70), element_justification="c")

    while True:
        event, values = window.read()

        if event == "-SUBMIT-":
            current_num = []
            current_num.append(values['-INPUT-'])
            num_string = ''.join(current_num)
            FreqInput = float(num_string)
            window['-TEXT-'].update(num_string)
            decoder()

        if event == "Back" or event == sg.WIN_CLOSED:
            break

    window.close()

def FRS():
    global BandSelect, FreqInput
    BandSelect = 3
    button_size = (11,3)
    sg.theme('DarkGrey6')
    sg.set_options(font = 'Franklin 16')
    layout = [
        [sg.Text("Select a Channel", font = 'Franklin 20')],
        [sg.Button("Channel 1", size = button_size), sg.Button("Channel 2", size = button_size), sg.Button("Channel 3", size = button_size),
            sg.Button("Channel 4", size = button_size), sg.Button("Channel 5", size = button_size), sg.Button("Channel 6", size = button_size)],
        [sg.Button("Channel 7", size = button_size), sg.Button("Channel 8", size = button_size), sg.Button("Channel 9", size = button_size),
            sg.Button("Channel 10", size = button_size), sg.Button("Channel 11", size = button_size), sg.Button("Channel 12", size = button_size)],
        [sg.Button("Channel 13", size = button_size), sg.Button("Channel 14", size = button_size), sg.Button("Channel 15", size = button_size),
            sg.Button("Channel 16", size = button_size), sg.Button("Channel 17", size = button_size), sg.Button("Channel 18", size = button_size)],
        [sg.Button("Channel 19", size = button_size), sg.Button("Channel 20", size = button_size), sg.Button("Channel 21", size = button_size),
            sg.Button("Channel 22", size = button_size)],
        [sg.Text("")],
        [sg.Text("Now playing:", font= 'Franklin 20'), sg.Text(key = '-TEXT-')],
        [sg.Button("Back", font = 'Franklin 20', size = (20,1))]
    ]

    window = sg.Window("FRS Decoder", layout, size=(1000,600), margins=(20,10), element_justification="c")

    while True:
        event, values = window.read()

        if event == "Channel 1":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.56250
            decoder()

        if event == "Channel 2":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.58750
            decoder()

        if event == "Channel 3":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.61250
            decoder()

        if event == "Channel 4":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.63750
            decoder()

        if event == "Channel 5":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.66250
            decoder()

        if event == "Channel 6":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.68750
            decoder()

        if event == "Channel 7":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.71250
            decoder()

        if event == "Channel 8":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.56250
            decoder()

        if event == "Channel 9":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.58750
            decoder()

        if event == "Channel 10":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.61250
            decoder()

        if event == "Channel 11":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.63750
            decoder()

        if event == "Channel 12":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.66250
            decoder()

        if event == "Channel 13":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.68750
            decoder()

        if event == "Channel 14":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 467.71250
            decoder()

        if event == "Channel 15":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.55000
            decoder()

        if event == "Channel 16":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.57500
            decoder()

        if event == "Channel 17":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.60000
            decoder()

        if event == "Channel 18":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.62500
            decoder()

        if event == "Channel 19":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.65000
            decoder()

        if event == "Channel 20":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.67500
            decoder()

        if event == "Channel 21":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.70000
            decoder()

        if event == "Channel 22":
            current_num = []
            current_num.append(event)
            num_string = ''.join(current_num)
            window['-TEXT-'].update(num_string)
            FreqInput = 462.72500
            decoder()

        if event == "Back" or event == sg.WIN_CLOSED:
            break

    window.close()

def main():
    sg.theme('DarkGrey6')
    button_size = (50,3)
    sg.set_options(font = 'Franklin 20')
    layout = [
        [sg.Text("Pick a mode of signal decoding")],
        [sg.Button("FM Radio", size = button_size)],
        [sg.Button("Weather Radio", size = button_size)],
        [sg.Button("FRS", size = button_size)],
        [sg.Text("")],
        [sg.Button("Exit", size = (25,1))]
    ]

    window = sg.Window("SDR Signal Decoder", layout, size=(1000,600), margins=(20, 10), element_justification="c")

    while True:
        event, values = window.read()
    
        if event == "FM Radio":
            FM_window()

        if event == "Weather Radio":
            Weather_Radio()

        if event == "FRS":
            FRS()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()

if __name__ == "__main__":
    main()

