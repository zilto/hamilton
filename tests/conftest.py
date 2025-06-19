from hamilton import registry, telemetry

if not registry.INITIALIZED:
    registry.initialize()

# disable telemetry for all tests!
telemetry.disable_telemetry()
