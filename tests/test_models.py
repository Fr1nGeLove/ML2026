import torch

from power_forecast.models import build_model


def test_all_models_return_horizon_vector() -> None:
    batch, input_days, features, horizon = 2, 90, 22, 365
    x = torch.randn(batch, input_days, features)

    for model_name in ["lstm", "transformer", "patch_channel_mixer", "trend_conv_transformer"]:
        model = build_model(
            model_name=model_name,
            input_dim=features,
            horizon=horizon,
            d_model=16,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        y = model(x)
        assert y.shape == (batch, horizon)
