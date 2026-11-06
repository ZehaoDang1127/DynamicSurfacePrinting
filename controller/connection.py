"""Explicit printer backend selection; a failed local connection never falls back to a network."""


def create_printer_client(config):
    if config.hardware.backend == "klipper_socket":
        from .klipper_socket import KlipperSocketClient
        return KlipperSocketClient(config)
    from .moonraker import MoonrakerClient
    return MoonrakerClient(config)
