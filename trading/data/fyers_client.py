from fyers_apiv3 import fyersModel

class FyersClient:
    def __init__(self, client_id: str, access_token: str):
        self.fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            is_async=False,
            log_path=""
        )

    def history(self, symbol: str, resolution: str, range_from: str, range_to: str):
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        return self.fyers.history(data)
