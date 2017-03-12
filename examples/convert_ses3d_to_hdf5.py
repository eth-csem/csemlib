from csemlib.models.ses3d import Ses3d

ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', ['rho', 'vsh', 'vsv', 'vph', 'vpv'])
ses3d.read()
ses3d.write_to_hdf5('/Users/Andreas/CSEM/csemlib/regional_models/japan/japan_new.hdf5')
