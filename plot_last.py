import pysmurf.client
import numpy as np
import scipy.signal as signal
import sys,os,glob
import argparse
import subprocess
from plotly.graph_objs import Scatter, Layout
from plotly import tools
import plotly.graph_objs as go
import plotly
from plotly import  subplots
from multiprocessing import Pool
import contextlib

freq_scaling = 95167.68567947495

def plotly_png(fig, name):
    fig.update_layout(paper_bgcolor='white')
    fig.write_image(name,width = 1960/1.5, height = 1080/1.5, scale = 1.5)

def plot_buffer(time, data, header, freq_mask, plot_filename, auto_open = False, cryocard_trace = None, decimation = None):
    '''

    '''
    print("Plotting buffer")
    #

    # Plotting
    fig = plotly.subplots.make_subplots(
        rows=1, cols=1,
        subplot_titles=('SMuRF datastream',),
        shared_xaxes=False,
        specs=[[{"secondary_y": True}]]
    )
    fig['layout']['yaxis1'].update(title='Df [Hz]')
    fig['layout']['xaxis1'].update(title='Time [s]')
    if decimation is not None:
        print("Decimating...")
        xx = signal.decimate((time- time[0])/1e9, int(decimation), ftype = 'fir')
        yy = signal.decimate(data, int(decimation), ftype = 'fir', axis = 1)
        hh = signal.decimate(cryocard_trace, int(decimation), ftype = 'fir')
    else:
        xx = (time- time[0])/1e9
        yy = data
        hh = cryocard_trace
    print("Plotting...")
    for ch in range(len(data)):
        fig.add_trace(go.Scatter(
                        x=xx[20:-20],
                        y=(yy[ch] -yy[ch][0])[20:-20] ,
                        name = "Ch %.1f MHz" % (freq_mask[ch]),
                        # showlegend=True,
                        # line={'color':'black'},
                        mode='lines',

                    ), secondary_y=False)
    if cryocard_trace is not None:
        fig.add_trace(go.Scatter(
                        x=xx[20:-20],
                        y=hh[20:-20],
                        name = "Cryocard trace",
                        # showlegend=True,
                        # line={'color':'black'},
                        mode='lines',

                    ),  secondary_y=True)
    # save to file
    fig['layout'].update(title="Plot of %s" % filename)
    plotly_png(fig, plot_filename+".png")
    plotly.offline.plot(fig, filename=plot_filename+".html", auto_open=auto_open)

def load_calib(filename):
    ctime = filename[:-4]+"_freq.txt"
    freq_mask = np.atleast_1d(np.loadtxt(ctime))

    # convert from streamed data to hz
    coef=1/240.
    coef_fixed = np.round(coef*2**15)
    filter_gain=coef_fixed*240*2**-15
    s2hz=1.2e6/np.pi/2/2/filter_gain

    return freq_mask, s2hz


def plot_NEP(samples, responsivity1, responsivity2, freq_mask, reso1, reso2, sampling_rate=1, welch=None, auto_open = False, filename = "SMuRF NEP SLAC"):
    '''
    Calculate pairdiff and pairsum specs, apply responsivity, plot the results

    Arguments:
        - Samples: array representing samples.
        - sampling_rate: sampling_rate
        - welch: in how many segment to divide the samples given for applying the Welch method
        - responsivity1/2: responsivity of each channel
        - reso1/2 : frequencies in MHz of each active channel (259)(251)
    Returns:
        None
    '''


    if welch == None:
        welch = len(samples[0])
    else:
        welch = int(len(samples[0]) / welch)

    for i in range(len(samples)):
        if np.abs(freq_mask[i] - reso1)<1:
            A = samples[i]
        elif np.abs(freq_mask[i] - reso2)<1:
            B = samples[i]

    DIFF = (A - B) *  responsivity
    SUM = (A + B) *  responsivity

    Frequencies, DSpec = signal.welch(DIFF, nperseg=welch, fs=sampling_rate, detrend='linear',scaling='density')
    Frequencies, SSpec = signal.welch(SUM, nperseg=welch, fs=sampling_rate, detrend='linear',scaling='density')

    DSpec = np.sqrt(DSpec)
    SSpec = np.sqrt(SSpec)
    # Plotting
    fig = plotly.subplots.make_subplots(
        rows=1, cols=1,
        subplot_titles=('SMuRF datastream',),
        shared_xaxes=False)
    fig['layout']['yaxis1'].update(title='NEP [aW/rt(Hz)]')
    fig['layout']['xaxis1'].update(title='Freq [Hz]')
    fig['layout'].update(xaxis_type="log")
    fig['layout'].update(yaxis_type="log")
    fig.append_trace(go.Scatter(
                    x=Frequencies,
                    y=DSpec,
                    name = "Pair Diff" ,
                    line={'color':'black'},
                    mode='lines'
                ), 1, 1)

    fig.append_trace(go.Scatter(
                    x=Frequencies,
                    y=SSpec,
                    name = "Pair Sum" ,
                    line={'color':'red'},
                    mode='lines'
                ), 1, 1)

    # save to file

    plot_filename = "plot/NEP_spec"
    fig['layout'].update(title="Plot of %s" % filename)
    plotly_png(fig, plot_filename+".png")
    plotly.offline.plot(fig, filename=plot_filename+".html", auto_open=auto_open)
    return Frequencies, SSpec, DSpec


def spec_from_samples(samples, freq_mask, sampling_rate=1, welch=None, auto_open = False):
    '''
    Calculate real and imaginary part of the spectra of a complex array using the Welch method.

    Arguments:
        - Samples: array representing samples.
        - sampling_rate: sampling_rate
        - welch: in how many segment to divide the samples given for applying the Welch method
    Returns:
        - Frequency array,
        - spectrum array
    '''


    if welch == None:
        welch = len(samples[0])
    else:
        welch = int(len(samples[0]) / welch)

    Frequencies, Spec = signal.welch(samples, nperseg=welch, fs=sampling_rate, detrend='linear',scaling='density')


    # Plotting
    fig = plotly.subplots.make_subplots(
        rows=1, cols=1,
        subplot_titles=('SMuRF datastream',),
        shared_xaxes=False)
    fig['layout']['yaxis1'].update(title='Df [Hz/rt(Hz)]')
    fig['layout']['xaxis1'].update(title='Freq [Hz]')
    fig['layout'].update(xaxis_type="log")
    fig['layout'].update(yaxis_type="log")
    for ch in range(len(Spec)):
        fig.append_trace(go.Scatter(
                        x=Frequencies,
                        y=Spec[ch],
                        name = "Ch %.1f MHz" % (freq_mask[ch]),
                        # showlegend=True,
                        # line={'color':'black'},
                        mode='lines'
                    ), 1, 1)

    # save to file
    filename = "test"
    plot_filename = "plot/test_spec"
    fig['layout'].update(title="Plot of %s" % filename)
    plotly_png(fig, plot_filename+".png")
    plotly.offline.plot(fig, filename=plot_filename+".html", auto_open=auto_open)

    return Frequencies, Spec

def read_file(filename_full):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        t,d,m,h = S.read_stream_data("%s" % (filename_full),return_header=True)
    return [t,d,m,h]


def get_data(filename):
    '''
    Get data from smurf server, cache it and returns time axis in seconds,
    data in Hz, mask, heater bias trace 1, and the frequency/channel lookup.
    '''

    save_name = os.path.basename(filename)[:-4]
    cache_present = glob.glob("data/"+save_name+"*")
    print(len(cache_present))
    Nfiles = glob.glob(filename+".*")
    Nfiles = len(Nfiles)
    if len(cache_present)<6:
        S = pysmurf.client.SmurfControl(offline=True)
        idxs=range(1,Nfiles+1)
        print(Nfiles)
        filenmes_full = ["%s.%d" % (filename,ii) for ii in idxs]
        with Pool(4) as p:
            m_list = p.map(read_file, [row for row in filenmes_full])

        print(filenmes_full)
        t=np.concatenate([m_list[ii][0] for ii in range(len(m_list))])
        d=np.concatenate([m_list[ii][1] for ii in range(len(m_list))],axis=1)
        m=np.concatenate([m_list[ii][2] for ii in range(len(m_list))],axis=1)
        h=np.concatenate([m_list[ii][3]['tes_bias'][0] for ii in range(len(m_list))])
        freq_mask, s2hz = load_calib(filename)

        np.save("data/"+save_name+"_time", t)
        np.save("data/"+save_name+"_data", d)
        np.save("data/"+save_name+"_mask", m)
        np.save("data/"+save_name+"_head", h)
        np.save("data/"+save_name+"_freq_mask", freq_mask)
        np.save("data/"+save_name+"_s2hz", s2hz)
    else:
        print("getting cached data")
        t = np.load("data/"+save_name+"_time.npy")
        d = np.load("data/"+save_name+"_data.npy")
        m = np.load("data/"+save_name+"_mask.npy")
        h = np.load("data/"+save_name+"_head.npy", allow_pickle=True)
        freq_mask= np.load("data/"+save_name+"_freq_mask.npy")
        s2hz= np.load("data/"+save_name+"_s2hz.npy")

    d*=s2hz

    return t*1e-9,d,m,h,freq_mask

def calculate_responsivity(time_ax, data, reso1, reso2, freq_mask, plot_filename, cryocard_trace = None, auto_open = False):
    '''
    Calculate the responsivity per each channel, returns responsivity1,responsivity2.
    Also plots the calibration measurement.
    '''

    for i in range(len(data)):
        if np.abs(freq_mask[i] - reso1)<1:
            A = data[i]
            A_freq = freq_mask[i]
        elif np.abs(freq_mask[i] - reso2)<1:
            B = data[i]
            B_freq = freq_mask[i]

    # remove fluctuations
    resampling_factor = len(A)/1000
    # A_f = signal.resample(signal.decimate(A, int(resampling_factor), ftype = 'fir'),len(A))
    # B_f = signal.resample(signal.decimate(B, int(resampling_factor), ftype = 'fir'),len(B))
    A_d = signal.decimate(A, int(resampling_factor), ftype = 'fir')
    B_d = signal.decimate(B, int(resampling_factor), ftype = 'fir')
    time_ax_d = signal.decimate(time_ax, int(resampling_factor), ftype = 'fir')

    A_coeff = np.polyfit(time_ax_d, A_d, 5, rcond=None, full=False, w=None, cov=False)
    p_A = np.poly1d(A_coeff)
    A_f = p_A(time_ax)

    B_coeff = np.polyfit(time_ax_d, B_d, 5, rcond=None, full=False, w=None, cov=False)
    p_B = np.poly1d(B_coeff)
    B_f = p_B(time_ax)

    #clipping
    clipping = 100
    B = B[clipping:-clipping]
    B_f = B_f[clipping:-clipping]
    A_d = A_d[clipping:-clipping]
    B_d = B_d[clipping:-clipping]
    time_ax_d = time_ax_d[clipping:-clipping]
    A_f = A_f [clipping:-clipping]
    A = A[clipping:-clipping]
    time_ax = time_ax[clipping:-clipping]

    print("Plotting...")
    fig = plotly.subplots.make_subplots(
        rows=1, cols=1,
        subplot_titles=('Calibration datastream',),
        shared_xaxes=False,
        specs=[[{"secondary_y": True}]]
    )
    fig['layout']['yaxis1'].update(title='Df [Hz]')
    fig['layout']['xaxis1'].update(title='Time [s]')
    fig.add_trace(go.Scatter(
                    x=time_ax_d,
                    y=A_d ,
                    name = "Ch %.1f MHz" % (A_freq),
                    # showlegend=True,
                    line={'color':'black'},
                    mode='lines',

                ), secondary_y=False)
    fig.add_trace(go.Scatter(
                    x=time_ax,
                    y=A_f ,
                    name = "resampled Ch %.1f MHz" % (A_freq),
                    # showlegend=True,
                    line={'color':'black'},
                    mode='lines',

                ), secondary_y=False)
    # fig.add_trace(go.Scatter(
    #                 x=time_ax,
    #                 y=B ,
    #                 name = "Ch %.1f MHz" % (B_freq),
    #                 # showlegend=True,
    #                 line={'color':'red'},
    #                 mode='lines',
    #
    #             ), secondary_y=False)
    # fig.add_trace(go.Scatter(
    #                 x=time_ax[clipping:-clipping],
    #                 y=B_f ,
    #                 name = "resampled Ch %.1f MHz" % (B_freq),
    #                 # showlegend=True,
    #                 line={'color':'red'},
    #                 mode='lines',
    #
    #             ), secondary_y=False)
    if cryocard_trace is not None:
        fig.add_trace(go.Scatter(
                        x=time_ax[clipping:-clipping],
                        y=cryocard_trace[clipping:-clipping],
                        name = "Cryocard trace",
                        # showlegend=True,
                        visible = 'legendonly',
                        # line={'color':'black'},
                        mode='lines',

                    ),  secondary_y=True)
    # save to file
    fig['layout'].update(title="Plot of %s" % plot_filename)
    plotly_png(fig, "plot/"+plot_filename+".png")
    plotly.offline.plot(fig, filename="plot/"+plot_filename+".html", auto_open=auto_open)

    return 0,0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Given a calibration file and a noise measurement, calculates the responsivity and generate pairdiff/sum NEP spec plot'
    )

    parser.add_argument('--calibration_filename', '-cf', help='Calibration measurement path', type=str)
    parser.add_argument('--noise_filename', '-nf', help='Noise measurement path', type=str)
    parser.add_argument('--reso1', '-r1', help='approximate frequency in MHz of the first biased resonator', type=float, default = 259)
    parser.add_argument('--reso2', '-r2', help='approximate frequency in MHz of the second biased resonator', type=float, default = 251)
    parser.add_argument('--sampling_rate', '-rate', help='Data acquisition rate', type=float, default = 500)
    parser.add_argument('--optical_power', '-power', help='Optical power used', type=float)

    args = parser.parse_args()

    if args.optical_power is None:
        print("--optical_power is required" )
        exit()

    S = pysmurf.client.SmurfControl(offline=True)
    save_name_calib = os.path.basename(args.calibration_filename)[:-4]

    t_calib,d_calib,m_calib,h_calib,freq_mask_calib = get_data(args.calibration_filename)
    # t_noise,d_noise,m_noise,h_noise,freq_mask_noise = get_data(args.noise_filename)

    responsivity1, responsivity2 = calculate_responsivity(
        time_ax = t_calib,
        data = d_calib,
        reso1 = args.reso1,
        reso2 = args.reso2,
        freq_mask = freq_mask_calib,
        plot_filename = "Calib_TKIDs_"+save_name_calib+"_%.1fpW"%(args.optical_power),
        cryocard_trace = h_calib
    )
    exit()

    plot_NEP(
        samples = d_noise,
        responsivity1 = responsivity1,
        responsivity2 = responsivity2,
        freq_mask = freq_mask_noise,
        reso1 = args.reso1,
        reso2 = args.reso2,
        sampling_rate=args.sampling_rate,
        welch=10,
        auto_open = False,
        filename = "NEP_TKIDs_"+save_name+"_%.1fpW"%(args.optical_power)
    )
