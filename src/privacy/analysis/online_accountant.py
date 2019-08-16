class OnlineAccountant(object):
    """ A class to perform accounting in an
    online manner to speed up experiments. requires
    an accountancy method to have an online method. """

    def __init__(self,
                 accountancy_update_method,
                 ledger=None,
                 accountancy_parameters=None):
        """
        :param accountancy_update_method: A method to compute the desired accountancy in and
        online fashion. Should take as parameters some list of new privacy queries to update the
        privacy for, and some tracking variable specific to the method (E.g. log moments for the
        moment accountant). This method should fuction if the tracking variable are None.
        :param ledger: Some initial ledger. May be None
        :param accountancy_parameters: Some parameters to pass to the accountancy update method.
        E.g. the target epsilon or delta, maximum log moment...
        """

        self._accountancy_update_method = accountancy_update_method
        self._accountancy_parameters = accountancy_parameters
        self._ledger = []
        self._tracking_parameters = None
        self._position = 0

        if ledger is None:
            ledger = []
        self._privacy_bound = self.update_privacy(ledger)

    def update_privacy(self, incremented_ledger):
        """ Update the current privacy bound using new additions to the ledger.

        :param incremented_ledger: The new ledger. Assumes that the only differences
        from previously seen ledger is the new entries. This should be of the formatted
        ledger type.
        :return: The new privacy bound.
        """
        self._ledger = incremented_ledger
        new_entries = self._ledger[self._position:]
        self._privacy_bound, self._tracking_parameters = self._accountancy_update_method(
            new_entries,
            self._tracking_parameters,
            **self._accountancy_parameters
        )
        self._position = len(self._ledger)
        return self._privacy_bound

    @property
    def privacy_bound(self):
        return self._privacy_bound
