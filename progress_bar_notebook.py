from __future__ import division
import datetime
import sys
import time
from ipywidgets import FloatProgress,Label,HBox,VBox,HTML
from IPython.display import display

from chainer.training import extension
from chainer.training import trigger

class ProgressBarNotebook(extension.Extension):

    """Trainer extension to print a progress bar and recent training status to jupyter notebook
    modified from original ProgressBar extension from chainer 1.13
    """
    def __init__(self, training_length=None, update_interval=100,):
        self._training_length = training_length
        self._status_template = None
        self._update_interval = update_interval
        self._recent_timing = []

        self.desc_total = Label("Total:")
        self.desc_total.layout.width="100px"
        self.pbar_total = FloatProgress(min=0, max=1.0,bar_style="success")
        self.text_total = Label("0%")
        self.desc_total.layout.padding="5px"
        self.text_total.layout.padding="5px"
        display( HBox([self.desc_total,self.pbar_total,self.text_total]))

        self.desc_epoch = Label("This epoch:")
        self.desc_epoch.layout.width="100px"
        self.pbar_epoch = FloatProgress(min=0, max=1.0)
        self.text_epoch = Label("0%")
        self.desc_epoch.layout.padding="5px"
        self.text_epoch.layout.padding="5px"
        display( HBox([self.desc_epoch,self.pbar_epoch,self.text_epoch]))

        self.epoch_report = Label("")
        self.time_report = Label("")
        self.epoch_report.layout.padding="5px"
        self.time_report.layout.padding="5px"
        display( VBox([self.epoch_report,self.time_report]))



    def __call__(self, trainer):
        training_length = self._training_length

        # initialize some attributes at the first call
        if training_length is None:
            t = trainer.stop_trigger
            if not isinstance(t, trigger.IntervalTrigger):
                raise TypeError(
                    'cannot retrieve the training length from %s' % type(t))
            training_length = self._training_length = t.period, t.unit

        stat_template = self._status_template
        if stat_template is None:
            stat_template = self._status_template = (
                '{0.iteration:10} iter, {0.epoch} epoch / %s %ss\n' %
                training_length)

        length, unit = training_length

        iteration = trainer.updater.iteration

        # print the progress bar
        if iteration % self._update_interval == 0:
            epoch = trainer.updater.epoch_detail
            recent_timing = self._recent_timing
            now = time.clock()

            if len(recent_timing) >= 1:

                if unit == 'iteration':
                    rate = iteration / length
                else:
                    rate = epoch / length

                self.pbar_total.value=rate
                self.text_total.value="{:6.2%}".format(rate)

                epoch_rate = epoch - int(epoch)

                self.pbar_epoch.value=epoch_rate
                self.text_epoch.value="{:6.2%}".format(epoch_rate)

                status = stat_template.format(trainer.updater)
                self.epoch_report.value=status

                old_t, old_e, old_sec = recent_timing[0]
                speed_t = (iteration - old_t) / (now - old_sec)
                speed_e = (epoch - old_e) / (now - old_sec)
                if unit == 'iteration':
                    estimated_time = (length - iteration) / speed_t
                else:
                    estimated_time = (length - epoch) / speed_e
                self.time_report.value = ('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                                        .format(speed_t,
                                          datetime.timedelta(seconds=estimated_time)))

                if len(recent_timing) > 100:
                    del recent_timing[0]

            recent_timing.append((iteration, epoch, now))

    def finalize(self):
        pass

