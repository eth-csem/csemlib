import pytest
import numpy as np

import csemlib.models.one_dimensional as m1d
DECIMAL_CLOSE = 3

def test_prem_no220():
    """
    Test to (mainly) make sure that discontinuities are handled properly.
    :return:
    """
    reg01 = np.array([3.374814283471982, 8.076866567257888, 8.076866567257888, 4.4708837074242656, 4.581983707424266,
                      0.0, 86.5042159786517, 57823.0])
    reg02 = np.array([3.363837607910846, 8.014433950714174, 8.014433950714174, 4.433659080207189, 4.433659080207189,
                      0.0, 143.00006327064492, 57823.0])
    reg03 = np.array([3.495466943964841, 8.751316606498193, 8.751316606498193, 4.7139374352534915, 4.7139374352534915,
                      0.0, 143.0, 57823.0])
    reg04 = np.array([3.5432636006906293, 8.905243242819024, 8.905243242819024, 4.7699,
                      4.7699, 0.0, 143.0, 57823.0])
    reg05 = np.array([3.7237469157118186, 9.133916669282687, 4.932487458797674, 9.133916669282687,
                      4.932487458797674, 0.0, 143.0, 57823.0])
    reg06 = np.array([3.9758203735677293, 10.157825003924035, 5.5159311881965145, 10.157825003924035,
                      5.5159311881965145, 0.0, 143.0, 57823.0])
    reg07 = np.array([3.9921213467273584, 10.266174462407786, 10.266174462407786, 5.570211034374509, 5.570211034374509,
                      0.0, 143.0, 57823.0])
    reg08 = np.array([4.3807429838542795, 10.751407424647873, 10.751407424647873, 5.945126361510368, 5.945126361510368,
                      0.0, 312.0, 57823.0])
    reg09 = np.array([4.443204194391242, 11.06568986974271, 11.06568986974271, 6.240535840301453, 6.240535840301453,
                      0.0, 312.0, 57823.0])
    reg10 = np.array([5.491476554415982, 13.680424477483925, 13.680424477483925, 7.2659252231153015, 7.2659252231153015,
                      0.0, 312.0, 57823.0])
    reg11 = np.array([5.566455445926154, 13.716622269026377, 13.716622269026377, 7.26465059504689, 7.26465059504689,
                      0.0, 312.0, 57823.0])
    reg12 = np.array([9.903438401183957, 8.064788053141768, 8.064788053141768, 0.0, 0.0, 0.0, 0.0, 57823.0])
    reg13 = np.array([12.166331854652926, 10.35571579802768, 10.35571579802768, 0.0, 0.0, 0.0, 0.0, 57823.0])
    reg14 = np.array([12.763614264456663, 11.02826139091006, 11.02826139091006, 3.5043113193074316, 3.5043113193074316,
                      0.0, 84.6, 1327.7])

    np.testing.assert_almost_equal(m1d.csem_1d_background(6292, region='upper_mantle'), reg01, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(6191, region='upper_mantle'), reg02, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(6051, region='upper_mantle'), reg03, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5971, region='upper_mantle'), reg04, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5971, region='transition_zone'), reg05, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5771, region='transition_zone'), reg06, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5701, region='transition_zone'), reg07, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5701, region='lower_mantle'), reg08, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(5600, region='lower_mantle'), reg09, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(3630, region='lower_mantle'), reg10, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(3480, region='lower_mantle'), reg11, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(3480, region='outer_core'), reg12, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(1221.5, region='outer_core'), reg13, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.csem_1d_background(1221.5, region='inner_core'), reg14, decimal=DECIMAL_CLOSE)

    # Make sure that questionable requests will error out.
    with pytest.raises(ValueError):
        m1d.csem_1d_background(6371, 'uppermantle')
    with pytest.raises(ValueError):
        m1d.csem_1d_background(5971, 'lower_mantle')
