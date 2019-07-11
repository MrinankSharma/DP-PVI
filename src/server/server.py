from abc import ABC, abstractmethod

class ParameterServer(ABC):
    def __init__(self, workers=[]):
        self.set_workers(workers)

    @abstractmethod
    def tick(self):
        '''
        Defines what the Parameter Server should do on each update round. Could be all worker synchronous updates,
        async updates, might check for new workers etc
        '''
        pass

    @abstractmethod
    def should_stop(self):
        '''
        Defines when the Parameter server should stop running. Might be a privacy limit, or anything else.'''
        pass

    def set_workers(self, workers):
        self.workers = workers

    def add_worker(self, worker):
        self.workers.append(worker)
