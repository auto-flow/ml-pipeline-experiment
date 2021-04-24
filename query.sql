select count(*) from trial;

select
       all_score ->> 'accuracy' as accuracy,
--        all_score ->> 'roc_auc_weighted' as auc,
--        all_score ->> 'f1' as f1,
--        all_score ->> 'balanced_accuracy' as ba,
--        all_score ->> 'mcc' as mcc,
       config ->> 'learner'     as learner,
       config ->> 'selector'    as selector,
       config ->> 'scaler'      as scaler
from trial
-- where config -> 'scaler' ->> 'StandardScaler' is not null
order by all_score ->> 'accuracy'  desc
limit 100;

select * from trial where failed_info is not null;

select count(*),config->>'scaler' as scaler from trial group by config->>'scaler';

