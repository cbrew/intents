# Intents

This repository demonstrates that for at least one intent classification dataset (banking77), the pretraining available in a widely distributed transformer 
model gets within ~ 1.7% of the accuracy available with extensive fine-tuning. The practical convenience of being able to use a community model 
very likely outweighs the benefit of the extra accuracy that is obtained from the fine-tuning process. It is probably worth trying 
the approach used here whenever attempting a new intent classification task.

The banking77 dataset is a standard benchmark for intent classification in dialogue. The strongest 
result of which we are currently aware, achieved with substantial effort, is accuracy=94.35, from models
that PolyAI currently chooses not to distribute.

```
@article{Vulic2021ConvFiTCF,
  title={ConvFiT: Conversational Fine-Tuning of Pretrained Language Models},
  author={Ivan Vulic and Pei-hao Su and Sam Coope and Daniel Gerz and Paweł Budzianowski and I{\~n}igo Casanueva and Nikola Mrkvsi'c and Tsung-Hsien Wen},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.10126}
}
```
In the course of a more complete investigation that my former employer (LivePerson) may in due
course publish, I discovered that using Huggingface's `all-mpnet-base-v2`, when used
with k-nearest neighbors, produces decent 
results (accuracy=92.70) *without any task-specific fine-tuning at all*. This is presumably because the model was pretrained
against highly compatible materials See [Huggingface's model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v1) 
for details of the pretraining process used there.



```                                                  precision    recall  f1-score   support

                                activate_my_card    0.95122   0.97500   0.96296        40
                                       age_limit    1.00000   1.00000   1.00000        40
                         apple_pay_or_google_pay    0.97561   1.00000   0.98765        40
                                     atm_support    0.97561   1.00000   0.98765        40
                                automatic_top_up    0.97561   1.00000   0.98765        40
         balance_not_updated_after_bank_transfer    0.76744   0.82500   0.79518        40
balance_not_updated_after_cheque_or_cash_deposit    1.00000   0.97500   0.98734        40
                         beneficiary_not_allowed    0.94872   0.92500   0.93671        40
                                 cancel_transfer    1.00000   1.00000   1.00000        40
                            card_about_to_expire    0.97561   1.00000   0.98765        40
                                 card_acceptance    0.92683   0.95000   0.93827        40
                                    card_arrival    0.82927   0.85000   0.83951        40
                          card_delivery_estimate    0.89744   0.87500   0.88608        40
                                    card_linking    1.00000   0.95000   0.97436        40
                                card_not_working    0.90476   0.95000   0.92683        40
                        card_payment_fee_charged    0.86047   0.92500   0.89157        40
                     card_payment_not_recognised    0.89189   0.82500   0.85714        40
                card_payment_wrong_exchange_rate    0.86047   0.92500   0.89157        40
                                  card_swallowed    1.00000   1.00000   1.00000        40
                          cash_withdrawal_charge    0.95000   0.95000   0.95000        40
                  cash_withdrawal_not_recognised    0.79592   0.97500   0.87640        40
                                      change_pin    0.95238   1.00000   0.97561        40
                                compromised_card    0.83784   0.77500   0.80519        40
                         contactless_not_working    1.00000   0.90000   0.94737        40
                                 country_support    0.97561   1.00000   0.98765        40
                           declined_card_payment    0.88636   0.97500   0.92857        40
                        declined_cash_withdrawal    0.88636   0.97500   0.92857        40
                               declined_transfer    0.94444   0.85000   0.89474        40
             direct_debit_payment_not_recognised    0.87179   0.85000   0.86076        40
                          disposable_card_limits    0.92857   0.97500   0.95122        40
                           edit_personal_details    1.00000   1.00000   1.00000        40
                                 exchange_charge    0.97436   0.95000   0.96203        40
                                   exchange_rate    0.88372   0.95000   0.91566        40
                                exchange_via_app    0.84091   0.92500   0.88095        40
                       extra_charge_on_statement    0.92683   0.95000   0.93827        40
                                 failed_transfer    0.86364   0.95000   0.90476        40
                           fiat_currency_support    0.89189   0.82500   0.85714        40
                     get_disposable_virtual_card    0.94595   0.87500   0.90909        40
                               get_physical_card    0.95122   0.97500   0.96296        40
                              getting_spare_card    0.95122   0.97500   0.96296        40
                            getting_virtual_card    0.90476   0.95000   0.92683        40
                             lost_or_stolen_card    0.85714   0.90000   0.87805        40
                            lost_or_stolen_phone    0.97561   1.00000   0.98765        40
                             order_physical_card    0.92500   0.92500   0.92500        40
                              passcode_forgotten    1.00000   1.00000   1.00000        40
                            pending_card_payment    0.86047   0.92500   0.89157        40
                         pending_cash_withdrawal    1.00000   0.90000   0.94737        40
                                  pending_top_up    0.88095   0.92500   0.90244        40
                                pending_transfer    0.90323   0.70000   0.78873        40
                                     pin_blocked    0.94444   0.85000   0.89474        40
                                 receiving_money    0.97222   0.87500   0.92105        40
                           Refund_not_showing_up    1.00000   1.00000   1.00000        40
                                  request_refund    1.00000   0.97500   0.98734        40
                          reverted_card_payment?    0.97500   0.97500   0.97500        40
                  supported_cards_and_currencies    0.86364   0.95000   0.90476        40
                               terminate_account    0.97561   1.00000   0.98765        40
                  top_up_by_bank_transfer_charge    0.88095   0.92500   0.90244        40
                           top_up_by_card_charge    0.97297   0.90000   0.93506        40
                        top_up_by_cash_or_cheque    0.97222   0.87500   0.92105        40
                                   top_up_failed    0.84615   0.82500   0.83544        40
                                   top_up_limits    0.97436   0.95000   0.96203        40
                                 top_up_reverted    0.86842   0.82500   0.84615        40
                              topping_up_by_card    0.84211   0.80000   0.82051        40
                       transaction_charged_twice    0.95238   1.00000   0.97561        40
                            transfer_fee_charged    0.91892   0.85000   0.88312        40
                           transfer_into_account    0.80435   0.92500   0.86047        40
              transfer_not_received_by_recipient    0.86486   0.80000   0.83117        40
                                 transfer_timing    0.92500   0.92500   0.92500        40
                       unable_to_verify_identity    0.92857   0.97500   0.95122        40
                              verify_my_identity    0.94595   0.87500   0.90909        40
                          verify_source_of_funds    0.97500   0.97500   0.97500        40
                                   verify_top_up    1.00000   1.00000   1.00000        40
                        virtual_card_not_working    1.00000   0.92500   0.96104        40
                              visa_or_mastercard    1.00000   0.95000   0.97436        40
                             why_verify_identity    0.92857   0.97500   0.95122        40
                   wrong_amount_of_cash_received    0.97222   0.87500   0.92105        40
         wrong_exchange_rate_for_cash_withdrawal    0.89189   0.82500   0.85714        40

                                        accuracy                        0.92695      3080
                                       macro avg    0.92861   0.92695   0.92668      3080
                                    weighted avg    0.92861   0.92695   0.92668      3080
```

A more detailed investigation reveals that several sentence transformer models have essentially the same
property as `all-mpnet-base-v2`, providing good embeddings for banking77 sentences out-of-the-box. The corresponding
base models (which are well documented at Huggingface), do not.

Default models:

- sentence-transformers/all-MiniLM-L6-v2 
  - nreimers/MiniLM-L6-H384-uncased (base model for above)
- sentence-transformers/all-mpnet-base-v2
  - microsoft/mpnet-base (base model for above)
- sentence-transformers/all-distilroberta-v1
  - distilroberta-base (base model for above)
- sentence-transformers/all-MiniLM-L12-v2
  - microsoft/MiniLM-L12-H384-uncased (base model for above)
- sentence-transformers/all-roberta-large-v1
  - roberta-large (base model for above)


Extra models: 

- sentence-transformers/multi-qa-mpnet-base-dot-v1  (base is mpnet)
- sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (base is mpnet)
- sentence-transformers/multi-qa-distilbert-cos-v1 (base is distilbert)
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (base is MiniLM-L12)