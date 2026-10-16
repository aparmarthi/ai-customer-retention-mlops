export type FeatureKind = 'number' | 'date' | 'gender';

export interface FeatureDef {
  key: string;
  label: string;
  kind: FeatureKind;
}

export interface FeatureGroup {
  id: 'profile' | 'billing' | 'engagement';
  title: string;
  features: FeatureDef[];
}

const n = (key: string, label: string): FeatureDef => ({ key, label, kind: 'number' });
const d = (key: string, label: string): FeatureDef => ({ key, label, kind: 'date' });

export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    id: 'profile',
    title: 'Profile',
    features: [
      n('city', 'City code'),
      n('bd', 'Age (bd)'),
      { key: 'gender', label: 'Gender', kind: 'gender' },
      n('registered_via', 'Registered via'),
      d('registration_init_time', 'Registration date'),
    ],
  },
  {
    id: 'billing',
    title: 'Billing',
    features: [
      n('txn_cnt', 'Transactions'),
      n('cancel_cnt', 'Cancellations'),
      n('cancel_rate', 'Cancel rate'),
      n('auto_renew_cnt', 'Auto-renew count'),
      n('auto_renew_rate', 'Auto-renew rate'),
      n('plan_list_price_mean', 'List price (mean)'),
      n('plan_list_price_max', 'List price (max)'),
      n('plan_list_price_min', 'List price (min)'),
      n('actual_paid_mean', 'Paid (mean)'),
      n('actual_paid_max', 'Paid (max)'),
      n('actual_paid_min', 'Paid (min)'),
      n('payment_method_nunique', 'Payment methods'),
      d('txn_first_date', 'First transaction'),
      n('txn_tenure_days_approx', 'Tenure days (approx)'),
      d('membership_expire_date_max', 'Membership expires'),
    ],
  },
  {
    id: 'engagement',
    title: 'Engagement',
    features: [
      n('log_row_cnt', 'Log rows'),
      d('log_first_date', 'First listen'),
      d('log_last_date', 'Last listen'),
      n('log_active_days', 'Active days'),
      n('log_span_days_approx', 'Span days (approx)'),
      n('total_secs_sum', 'Total seconds'),
      n('total_secs_mean_per_active_day', 'Secs / active day'),
      n('total_secs_max_per_day', 'Max secs / day'),
      n('num_unq_sum', 'Unique tracks'),
      n('num_unq_mean_per_active_day', 'Unique / active day'),
      n('num_unq_max_per_day', 'Max unique / day'),
      n('num_25_sum', 'Plays < 25%'),
      n('num_50_sum', 'Plays 25-50%'),
      n('num_75_sum', 'Plays 50-75%'),
      n('num_985_sum', 'Plays 75-98.5%'),
      n('num_100_sum', 'Plays > 98.5%'),
      n('listen_full_share', 'Full-listen share'),
      n('listens_per_unique_track', 'Listens / unique track'),
    ],
  },
];

export const FEATURES: string[] = FEATURE_GROUPS.flatMap((g) => g.features.map((f) => f.key));

export const GENDERS = ['male', 'female', 'unknown'] as const;
