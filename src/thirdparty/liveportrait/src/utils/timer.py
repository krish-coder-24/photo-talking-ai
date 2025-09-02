# coding: utf-8

"""
tools to measure elapsed time
"""

import time

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        
        self.window = 100
        self.pre_len = 0
        self.times = []
        self.average = 0.
        self.first_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        
        self.diff = time.time() - self.start_time
        self.times.append(self.diff)
        
        if average:
            self.average = (self.diff + self.average * self.pre_len - self.first_time) / len(self.times)
        else:
            self.average = self.diff
            
        self.pre_len = len(self.times)
            
        if len(self.times) > self.window:
            self.first_time = self.times.pop(0)
        
        return self.average

    def clear(self):
        self.start_time = 0.
        self.diff = 0.
        self.times = []
        self.pre_len = 0
        self.average = 0.
        self.first_time = 0.
        
    def fout(self, secs):
        secs = int(secs)
        days = secs // (24 * 3600)
        remaining_secs = secs % (24 * 3600)
        hours = remaining_secs // 3600
        remaining_secs %= 3600
        minutes = remaining_secs // 60
        seconds = remaining_secs % 60
        
        return f"{days}d,{hours:02d}-{minutes:02d}-{seconds:02d}"
        
        
