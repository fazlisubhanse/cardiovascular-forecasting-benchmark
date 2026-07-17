from src.deep_learning.architectures import architecture_size_candidates,build_architecture
from src.deep_learning.parameter_audit import parameter_counts


def test_small_lstm_manual_parameter_count_and_capacity_order():
    assert parameter_counts(build_architecture("LSTM",{"hidden":8},0))["trainable_parameter_count"]==361
    compact=parameter_counts(build_architecture("CNN_BiLSTM_Attention_Compact",architecture_size_candidates("CNN_BiLSTM_Attention_Compact")[0],0))["trainable_parameter_count"]
    original=parameter_counts(build_architecture("CNN_BiLSTM_Attention_Original23K",architecture_size_candidates("CNN_BiLSTM_Attention_Original23K")[0],0))["trainable_parameter_count"]
    assert compact<original and 20000<original<25000
