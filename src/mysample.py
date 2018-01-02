import numpy as np
import tensorflow as tf
import pickle as pickle
import sys

from utils import *


def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def get_style_states(model, args):
    c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
    h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()
    if args.style is -1: return [c0, c1, c2, h0, h1, h2] #model 'chooses' random style
    with open(os.path.join(args.data_dir, 'styles.p'),'rb') as f:
        styles = pickle.load(f, encoding='latin1')
        style_strokes, style_strings = styles

    style_strokes, style_string = style_strokes[args.style], style_strings[args.style]
    style_onehot = [to_one_hot(style_string, model.ascii_steps, args.alphabet)]
        
    style_stroke = np.zeros((1, 1, 3), dtype=np.float32)
    style_kappa = np.zeros((1, args.kmixtures, 1))
    prime_len = 500 # must be <= 700
    
    for i in range(prime_len):
        style_stroke[0][0] = style_strokes[i,:]
        feed = {model.input_data: style_stroke, model.char_seq: style_onehot, model.init_kappa: style_kappa, \
                model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
                model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
        fetch = [model.new_kappa, \
                 model.fstate_cell0.c, model.fstate_cell1.c, model.fstate_cell2.c,
                 model.fstate_cell0.h, model.fstate_cell1.h, model.fstate_cell2.h]
        [style_kappa, c0, c1, c2, h0, h1, h2] = model.sess.run(fetch, feed)
    return [c0, c1, c2, np.zeros_like(h0), np.zeros_like(h1), np.zeros_like(h2)] #only the c vectors should be primed

def sample(input_text, model, args):
    # initialize some parameters
    one_hot = [to_one_hot(input_text, model.ascii_steps, args.alphabet)]         # convert input string to one-hot vector
    [c0, c1, c2, h0, h1, h2] = get_style_states(model, args) # get numpy zeros states for all three LSTMs
    kappa = np.zeros((1, args.kmixtures, 1))   # attention mechanism's read head should start at index 0
    prev_x = np.asarray([[[0, 0, 1]]], dtype=np.float32)     # start with a pen stroke at (0,0)

    strokes, windows, phis, kappas = [], [], [], [] # the data we're going to generate will go here

    finished = False ; i = 0
    while not finished:
        feed = {model.input_data: prev_x, model.char_seq: one_hot, model.init_kappa: kappa, \
                model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2, \
                model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}
        fetch = [model.x, model.y, model.eos, model.eos_softmax,\
                 model.window, model.phi, model.new_kappa, model.alpha, \
                 model.fstate_cell0.c, model.fstate_cell1.c, model.fstate_cell2.c,\
                 model.fstate_cell0.h, model.fstate_cell1.h, model.fstate_cell2.h]
        [x, y, eos, eos_softmax,  window, phi, kappa, alpha, \
                 c0, c1, c2, h0, h1, h2] = model.sess.run(fetch, feed)
        
        eos = [[0]]
        print("eos softmax", eos_softmax[0][0], eos_softmax[0][1])
        if eos_softmax[0][1]> 0.5:
            eos[0][0] = 1
        else:
            eos[0][0] = 0 # use 0.5 as arbitrary boundary
                
        # store the info at this time step
        windows.append(window)
        phis.append(phi[0])
        kappas.append(kappa[0].T)
        #pis.append(pi[0])
        strokes.append([x[0][0], y[0][0], eos[0][0]])
        
        # test if finished (has the read head seen the whole ascii sequence?)
        # main_kappa_idx = np.where(alpha[0]==np.max(alpha[0]));
        # finished = True if kappa[0][main_kappa_idx] > len(input_text) else False
        finished = True if i > args.tsteps else False
        
        # new input is previous output
        prev_x[0][0] = np.array([x, y, eos], dtype=np.float32).squeeze(axis=(1,2))
        i+=1

    windows = np.vstack(windows)
    phis = np.vstack(phis)
    kappas = np.vstack(kappas)
    strokes = np.vstack(strokes)

    # the network predicts the displacements between pen points, so do a running sum over the time dimension
    strokes[:,:2] = np.cumsum(strokes[:,:2], axis=0)
    return strokes, phis, windows, kappas


# plots parameters from the attention mechanism
def window_plots(phis, windows, save_path='.'):
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.title('Phis', fontsize=20)
    plt.xlabel("ascii #", fontsize=15)
    plt.ylabel("time steps", fontsize=15)
    plt.imshow(phis, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.subplot(122)
    plt.title('Soft attention window', fontsize=20)
    plt.xlabel("one-hot vector", fontsize=15)
    plt.ylabel("time steps", fontsize=15)
    plt.imshow(windows, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

# a heatmap for the probabilities of each pen point in the sequence
def gauss_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib.mlab as mlab
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize) #
    buff = 1 ; epsilon = 1e-4
    minx, maxx = np.min(strokes[:,0])-buff, np.max(strokes[:,0])+buff
    miny, maxy = np.min(strokes[:,1])-buff, np.max(strokes[:,1])+buff
    delta = abs(maxx-minx)/400. ;

    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(strokes.shape[0]):
        gauss = mlab.bivariate_normal(X, Y, mux=strokes[i,0], muy=strokes[i,1], \
            sigmax=strokes[i,2], sigmay=strokes[i,3], sigmaxy=0) # sigmaxy=strokes[i,4] gives error
        Z += gauss/(np.max(gauss) + epsilon)

    plt.title(title, fontsize=20)
    plt.imshow(Z)
    plt.savefig(save_path)
    plt.clf() ; plt.cla()

# plots the stroke data (handwriting!)
def line_plot(strokes, title, figsize = (20,2), save_path='.'):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    eos_preds = np.where(strokes[:,-1] == 1)
    eos_preds = [0] + list(eos_preds[0]) + [-1] #add start and end indices
    for i in range(len(eos_preds)-1):
        start = eos_preds[i]+1
        stop = eos_preds[i+1]
        plt.plot(strokes[start:stop,0], strokes[start:stop,1],'b-', linewidth=2.0) #draw a stroke
    plt.title(title,  fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.clf() ; plt.cla()
