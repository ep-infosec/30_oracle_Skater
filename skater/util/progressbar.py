# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import sys
import os


class ProgressBar(object):
    """Progress bar

    Parameters:

        n_cycles(int):
            The number of iterations.

        units(str):
            The unit name to be display (e.g. 10 iterations...)

    """
    def __init__(self, n_cycles, units='iterations'):

        self.start_time = time.time()
        self.n_cycles = n_cycles
        self.current_cycle = 0
        self.units = units
        self.todo_s = "-"
        self.complete_s = '█'
        self.bar_length = 20


    def printer(self, s):
        sys.stdout.write("\r{}".format(s))
        sys.stdout.flush()


    def format_count(self, x):
        """
        x is between 0 and 1
        if bar_length is 20, then complete % needs to .05
        """
        return int((x * 100) / 5)


    def animate(self):
        self.current_cycle += 1
        elapsed = int(time.time() - self.start_time)
        time_elapsed = " Time elapsed: {} seconds".format(elapsed)
        counter = "[{0}/{1}] {2} ".format(self.current_cycle, self.n_cycles, self.units)

        n_complete_to_display = self.format_count(self.current_cycle / float(self.n_cycles))
        n_todo_to_display = self.bar_length - n_complete_to_display
        bar = self.complete_s * n_complete_to_display + self.todo_s * n_todo_to_display
        msg = counter + bar + time_elapsed
        self.printer(msg)


# def is_run_from_ipython():
#     try:
#         __IPYTHON__
#         return True
#     except NameError:
#         return False
