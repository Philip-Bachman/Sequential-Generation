import matplotlib.pyplot as plt

########################################################
# plot results from tests of refinement process length #
########################################################
refine_steps = [1, 2, 5, 10, 15]
f = lambda nll: nll * (1.0 / 0.9)
add_scores = [f(87.75), f(81.51), f(77.56), f(75.81), f(75.48)]
jump_scores = [f(87.56), f(81.52), f(80.42), f(82.55), f(83.83)]
#
fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = fig.add_subplot(111)
ax.hold(True)
plt.title('The Effect of Increased Refinement Steps', fontsize=20)
ax.set_xlabel('Refinement Steps', fontsize=18)
ax.plot(refine_steps, add_scores, linewidth=2.0, marker=u'o', markersize=10, label='GPSI-add')
ax.plot(refine_steps, jump_scores, linewidth=2.0, marker=u'+', mew=2.0, markersize=12, label='GPSI-jump')
#
x_locs, x_labels = plt.xticks()
plt.xticks(x_locs, fontsize=18)
y_locs, y_labels = plt.yticks()
plt.yticks(y_locs, fontsize=18)
ax.legend(loc='upper left', fontsize=18)
fig.savefig('mcar_result_plot_3.pdf', dpi=None, facecolor='w', edgecolor='w',     orientation='portrait', papertype=None, format=None,     transparent=False, bbox_inches=None, pad_inches=0.1,     frameon=None)
plt.close(fig)


########################################################
# plot results from tests of more/less missing pixels  #
########################################################
drop_rates = [0.6, 0.7, 0.8, 0.9]
tm_orc = [165.0, 166.0, 167.0, 168.0]
tm_hon = [210.0, 220.0, 240.0, 302.0]
vae_imp = [105.0, 145.0, 190.0, 265.0]
gpsi_add = [74.5, 77.8, 80.5, 85.0]
gpsi_jump = [74.2, 77.2, 82.1, 86.1]
lstm_add = [71.7, 73.5, 75.9, 79.9]
lstm_jump = [71.8, 73.6, 76.2, 80.3]

#
# plot results from all models
#
fig = plt.figure()
ax = fig.add_subplot(111)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax.hold(True)
plt.title('Imputation NLL vs.~Available Information', fontsize=20)
ax.set_xlabel('Mask Probability', fontsize=18)
ax.plot(drop_rates, tm_orc, linewidth=2.0, marker=u'o', markersize=10, label='TM-orc')
ax.plot(drop_rates, tm_hon, linewidth=2.0, marker=u'+', mew=2.0, markersize=12, label='TM-hon')
ax.plot(drop_rates, vae_imp, linewidth=2.0, marker=u's', markersize=10, label='VAE-imp')
ax.plot(drop_rates, gpsi_add, linewidth=2.0, marker=u'x', mew=2.0, markersize=12, label='GPSI-add')
ax.plot(drop_rates, gpsi_jump, linewidth=2.0, marker=u'^', markersize=10, label='GPSI-jump')
ax.plot(drop_rates, lstm_add, linewidth=2.0, marker=u'*', markersize=12, label='LSTM-add')
ax.plot(drop_rates, lstm_jump, linewidth=2.0, marker=u'd', markersize=10, label='LSTM-jump')
#
plt.xlim(0.55, 0.95)
x_locs, x_labels = plt.xticks()
plt.xticks(x_locs, fontsize=18)
y_locs, y_labels = plt.yticks()
plt.yticks(y_locs, fontsize=18)
ax.legend(loc='upper left', fontsize=18)
fig.savefig('mcar_result_plot_1.pdf', dpi=None, facecolor='w', edgecolor='w',     orientation='portrait', papertype=None, format=None,     transparent=False, bbox_inches=None, pad_inches=0.1,     frameon=None)
plt.close(fig)

#
# plot results from our models
#
fig = plt.figure()
ax = fig.add_subplot(111)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax.hold(True)
plt.title('Imputation NLL vs.~Available Information', fontsize=20)
ax.set_xlabel('Mask Probability', fontsize=18)
ax.plot(drop_rates, gpsi_add, linewidth=2.0, marker=u'o', markersize=10, label='GPSI-add')
ax.plot(drop_rates, gpsi_jump, linewidth=2.0, marker=u'+', mew=2.0, markersize=12, label='GPSI-jump')
ax.plot(drop_rates, lstm_add, linewidth=2.0, marker=u's', markersize=10, label='LSTM-add')
ax.plot(drop_rates, lstm_jump, linewidth=2.0, marker=u'x', mew=2.0, markersize=12, label='LSTM-jump')
#
plt.xlim(0.55, 0.95)
x_locs, x_labels = plt.xticks()
plt.xticks(x_locs, fontsize=18)
y_locs, y_labels = plt.yticks()
plt.yticks(y_locs, fontsize=18)
ax.legend(loc='upper left', fontsize=18)
fig.savefig('mcar_result_plot_2.pdf', dpi=None, facecolor='w', edgecolor='w',     orientation='portrait', papertype=None, format=None,     transparent=False, bbox_inches=None, pad_inches=0.1,     frameon=None)
plt.close(fig)
