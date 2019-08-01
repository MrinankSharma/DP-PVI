import src.privacy_accounting.analysis.moment_accountant as ma
import src.privacy_accounting.analysis.pld_accountant as pld
import src.privacy_accounting.analysis.pld_accountant.compute_eps_var as pld2
from src.privacy_accounting.analysis import OnlineAccountant, PrivacyLedger

ledger = PrivacyLedger(1000, 0.01)

online_ma = OnlineAccountant(
    ma.compute_online_privacy_from_ledger,
    accountancy_parameters={
        'target_delta': 0.001
    }
)

online_pld = OnlineAccountant(
    pld.compute_online_privacy_from_ledger,
    accountancy_parameters={
        'target_delta': 0.001,
    }
)

for i in range(10):
    ledger.record_sum_query(5, 4)
    ledger.finalise_sample()

    # print('Moment Accountant', i, online_ma.update_privacy(ledger.get_formatted_ledger()))
    # print('PLD Accountant', i, online_pld.update_privacy(ledger.get_formatted_ledger()))

print('Moment Accountant', 'Final',
      ma.compute_privacy_loss_from_ledger(ledger.get_formatted_ledger(), target_delta=0.001))

for L in range(50, 1000, 200):
    print('PLD Accountant ar', 'Final', L,
          pld.compute_privacy_loss_from_ledger(ledger.get_formatted_ledger(), target_delta=0.001, L=L))
    print('PLD Accountant2 ar', 'Final', L,
          pld2.compute_privacy_loss_from_ledger(ledger.get_formatted_ledger(), target_delta=0.001, L=L))
    print('PLD Accountant sub', 'Final', L,
          pld.compute_privacy_loss_from_ledger(ledger.get_formatted_ledger(), target_delta=0.001, L=L,
                                               adjacency_definition="substitution"))
    print('PLD Accountant2 sub', 'Final', L,
          pld2.compute_privacy_loss_from_ledger(ledger.get_formatted_ledger(), target_delta=0.001, L=L,
                                                adjacency_definition="substitution"))
