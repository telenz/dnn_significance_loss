import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import plotly.graph_objs as go
import plotly.plotly as py
import significance_estimators as sig

color1  = '#53bab0'
color2  = '#fbc96d'
color3  = '#ffa147'
color4  = '#FFCC66'
color5  = "#DE5A6A";
color6  = "#03A8F5";
color7  = "#A8CCA4";
color8  = "#BEE6E7";
color9  = "#9999CC";
color10 = "#FFCCFF";
color11 = "#05B0BB"
color12 = "#03A8F5";
color13 = "#BB051E";
color14 = "#EA32E9";
color15 = "#CCFFCC";

plt.rcParams.update({'font.size': 20})

def plot_significances(df_test_with_pred, weight_name):
   
   df_sig = df_test_with_pred.loc[df_test_with_pred['signal']==1]
   df_bkg = df_test_with_pred.loc[df_test_with_pred['signal']==0]

   # Get histograms
   plt.figure(figsize=(15,8))
   plt.ylabel('Events')
   plt.xlabel('NN probability')
   plt.title('Classification Power')
   plt.yscale('log')
   n_bins = 250
   h_bkg = plt.hist(df_bkg['pred_prob'], n_bins, facecolor=color1, alpha=0.6, cumulative=-1, weights=df_bkg[weight_name])
   h_sig = plt.hist(df_sig['pred_prob'], n_bins, facecolor=color2, alpha=0.6, cumulative=-1, weights=df_sig[weight_name])
   plt.legend(['Background','Signal'])
   plt.savefig("plots/cumulative_classifier_plot.png")
   

   s, b = h_sig[0], h_bkg[0]
   bin_centers = (h_sig[1][:-1] + h_sig[1][1:])/2
   plt.figure(figsize=(15,8))
   n_bins_filled = min(sum(s>0), sum(b>0))
   s = s[:n_bins_filled]
   b = b[:n_bins_filled]
   bin_centers = bin_centers[:n_bins_filled]

   asimov_sys_0p01 = sig.asimov(0.01)
   asimov_sys_0p1  = sig.asimov(0.1)
   asimov_sys_0p3  = sig.asimov(0.3)
   asimov_sys_0p5  = sig.asimov(0.5)
   asimov_sys_0p1_with_reg  = sig.asimov_with_reg(0.1)
   asimov_sys_0p3_with_reg  = sig.asimov_with_reg(0.3)
   asimov_sys_0p5_with_reg  = sig.asimov_with_reg(0.5)

   plt.plot(bin_centers, sig.AMS(s,b),              color=color1, linewidth=2.0, label='AMS : ' + str(round(max(sig.AMS(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p01(s,b),      color=color4, linewidth=2.0, label='Z_asimov (sys=0.01) : ' + str(round(max(asimov_sys_0p01(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p1(s,b),       color=color3, linewidth=2.0, label='Z_asimov (sys=0.10) : ' + str(round(max(asimov_sys_0p1(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p3(s,b),       color=color5, linewidth=2.0, label='Z_asimov (sys=0.30) : ' + str(round(max(asimov_sys_0p3(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p5(s,b),       color=color9, linewidth=2.0, label='Z_asimov (sys=0.50) : ' + str(round(max(asimov_sys_0p5(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p1_with_reg(s,b),       color=color13, linewidth=2.0, label='Z_asimov_with_reg (sys=0.10) : ' + str(round(max(asimov_sys_0p1_with_reg(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p3_with_reg(s,b),       color=color14, linewidth=2.0, label='Z_asimov_with_reg (sys=0.30) : ' + str(round(max(asimov_sys_0p3_with_reg(s,b)),1)) )
   plt.plot(bin_centers, asimov_sys_0p5_with_reg(s,b),       color=color15, linewidth=2.0, label='Z_asimov_with_reg (sys=0.50) : ' + str(round(max(asimov_sys_0p5_with_reg(s,b)),1)) )
   plt.plot(bin_centers, sig.s_over_sqrt_of_b(s,b), color=color7, linewidth=2.0, dashes=[6, 2], label='s/sqrt(b) : ' + str(round(max(sig.s_over_sqrt_of_b(s,b)),1)) )
   plt.xlabel('NN probablity')
   plt.ylabel('Significance estimate')
   plt.legend(loc=2)
   plt.grid()

   plt.savefig("plots/significance_estimates.png")

   


def plot_prediction(df_test_with_pred):
    
   color1 = '#53bab0'
   color2 = '#fbc96d' #'#ffa147'

   # the histogram of the data
   df_sig = df_test_with_pred.loc[df_test_with_pred['signal']==1]
   df_bkg = df_test_with_pred.loc[df_test_with_pred['signal']==0]

   plt.figure(figsize=(15,8))
   plt.xlabel('NN probablity')
   plt.ylabel('Events')
   plt.yscale('log')
   plt.title('Classification Power')
   n, bins, patches = plt.hist(df_bkg['pred_prob'], 50, facecolor=color1, alpha=0.5, normed=True)
   n, bins, patches = plt.hist(df_sig['pred_prob'], 50, facecolor=color2, alpha=0.5, normed=True)

   plt.legend(['Background','Signal'])

   #plt.show()
   if not os.path.exists('plots'):
       os.makedirs('plots')
   plt.savefig("plots/classification.png")

def plot_val_train_loss(history, plot_log = True):

    color1 = '#adbc8a' #'#53bab0'
    color2 = '#fbc96d' #'#ffa147'

    #color1 = '#53bab0' # green
    #color2 = '#ffa147' # orange

    plt.figure(figsize=(15,8))

    # Get training and test loss histories
    training_loss   = history.history['loss']
    validation_loss = history.history['val_loss']
    epoch_range     = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_range, training_loss,   color1,  linewidth=4.0)
    plt.plot(epoch_range, validation_loss, color2,   linewidth=4.0)
    plt.legend(['Training Loss    : ' + str(round( min(history.history["loss"]) ,4)), 
                'Validation Loss : ' + str(round( min(history.history["val_loss"]) ,4)) ])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if plot_log: plt.yscale('log')
    plt.show(block=False);
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig("plots/loss.png")


def plot_val_train_loss_plotly(history):

    training_loss   = history.history['loss']
    validation_loss = history.history['val_loss'] # FIXME: 
    epoch_range     = range(1, len(training_loss) + 1)

    loss_train = go.Scatter(
        x = epoch_range,
        y = training_loss,
        mode = 'lines',
        name = 'training loss',
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 4,
            dash = 'dot')
        )
    
    loss_val = go.Scatter(
        x = epoch_range,
        y = validation_loss,
        mode = 'lines',
        name = 'validation loss',
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4)
        )
    
    data = [loss_train, loss_val]

    layout = dict(title = 'Loss vs. Epochs',
                  xaxis = dict(title = 'Epochs'),
                  yaxis = dict(title = 'Loss'),
                  )

    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='basic-line')

