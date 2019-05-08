import ptvsd
ptvsd.enable_attach(address=('0.0.0.0', 7102))
ptvsd.wait_for_attach()