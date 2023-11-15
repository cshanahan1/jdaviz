import warnings
import pytest

from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
import astropy.units as u
from glue.core.roi import EllipticalROI, CircularROI, XRangeROI
from glue_astronomy.translators.regions import roi_subset_state_to_region
from jdaviz.configs.imviz.helper import link_image_data
import numpy as np
from numpy.testing import assert_allclose


@pytest.mark.filterwarnings('ignore')
def test_plugin(specviz_helper, spectrum1d):
    specviz_helper.load_data(spectrum1d)
    p = specviz_helper.plugins['Subset Tools']

    # regression test for https://github.com/spacetelescope/jdaviz/issues/1693
    sv = specviz_helper.app.get_viewer('spectrum-viewer')
    sv.apply_roi(XRangeROI(6500, 7400))

    p._obj.subset_select.selected = 'Create New'

    po = specviz_helper.plugins['Plot Options']
    po.layer = 'Subset 1'
    po.line_color = 'green'


def test_subset_definition_with_composite_subset(cubeviz_helper, spectrum1d_cube):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cubeviz_helper.load_data(spectrum1d_cube)
    cubeviz_helper.app.get_tray_item_from_name('g-subset-plugin')

def test_circle_recenter_linking(imviz_helper, image_2d_wcs):
    arr = np.ones((10, 10))
    ndd = NDData(arr, wcs=image_2d_wcs)
    imviz_helper.load_data(ndd, data_label='dataset1')

    # force link to be pixel initially.
    link_image_data(imviz_helper.app, link_type='pixels')

    # apply circular subset
    imviz_helper.app.get_viewer('imviz-0').apply_roi(CircularROI(xc=5, yc=5, radius=2))

    # get plugin and check that attribute tracking link type is set properly
    plugin = imviz_helper.plugins['Subset Tools']._obj
    assert plugin.display_sky_coordinates is False

    # get initial subset definitions from ROI applied
    subset_defs = plugin.subset_definitions

    # check that the subset definitions, which control what is displayed in the UI, are correct
    true_vals = (('X Center (pixels)'), 5), ('Y Center (pixels)', 5), ('Radius (pixels)', 2)
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert subset_defs[0][i+1]['value'] == true_val[1]

    # get original subset location as a sky region for use later
    original_subs = imviz_helper.app.get_subsets(include_sky_region=True)
    original_sky_region = original_subs['Subset 1'][0]['sky_region']

    # move subset (subset state is what is modified in UI)
    plugin._set_value_in_subset_definition(0, 'X Center (pixels)', 'xc', 6)
    plugin._set_value_in_subset_definition(0, 'Y Center (pixels)', 'yc', 6)
    plugin._set_value_in_subset_definition(0, 'Radius (pixels)', 'radius', 3)

    # update subset to apply these changes
    plugin.vue_update_subset()
    subset_defs = plugin.subset_definitions

    # and check that it is changed after vue_update_subset runs
    true_vals = (('X Center (pixels)', 6), ('Y Center (pixels)', 6),
                 ('Radius (pixels)', 3))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert subset_defs[0][i+1]['value'] == true_val[1]

    # get oupdated subset location as a sky region
    updated_sky_region = imviz_helper.app.get_subsets(include_sky_region=True)
    updated_sky_region = updated_sky_region['Subset 1'][0]['sky_region']

    # remove subsets and change link type to wcs
    dc = imviz_helper.app.data_collection
    dc.remove_subset_group(dc.subset_groups[0])
    link_image_data(imviz_helper.app, link_type='wcs')
    assert plugin.display_sky_coordinates is True  # linking change should trigger this to change to True

    # apply subset
    img_wcs = imviz_helper.app.data_collection[1].data.coords
    # make sure pixel positions are in the same place as first test case when pixel linked
    x, y = img_wcs.world_to_pixel(SkyCoord(original_sky_region.center.ra,
                                           original_sky_region.center.dec))
    imviz_helper.app.get_viewer('imviz-0').apply_roi(CircularROI(xc=x, yc=y, radius=2.))

    # subset definition should now be in sky coordinates
    subset_defs = plugin.subset_definitions
    # check that the subset definitions, which control what is displayed in the UI, are correct
    true_vals = (('RA Center (degrees)', original_sky_region.center.ra.deg),
                 ('Dec Center (degrees)', original_sky_region.center.dec.deg),
                 ('Radius (degrees)', original_sky_region.radius.to(u.deg).value))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert_allclose(subset_defs[0][i+1]['value'], true_val[1])

    # move the to the same position as last time we updated it
    plugin._set_value_in_subset_definition(0, 'RA Center (degrees)', 'xc', updated_sky_region.center.ra.deg)
    plugin._set_value_in_subset_definition(0, 'Dec Center (degrees)', 'yc', updated_sky_region.center.dec.deg)
    plugin._set_value_in_subset_definition(0, 'Radius (degrees)', 'radius', updated_sky_region.radius.to(u.deg).value)

    # update subset
    plugin.vue_update_subset()

    subset_defs = plugin.subset_definitions
    true_vals = (('RA Center (degrees)', updated_sky_region.center.ra.deg),
                 ('Dec Center (degrees)', updated_sky_region.center.dec.deg),
                 ('Radius (degrees)', updated_sky_region.radius.to(u.deg).value))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert_allclose(subset_defs[0][i+1]['value'], true_val[1])


def test_ellipse_recenter_linking(imviz_helper, image_2d_wcs):
    arr = np.ones((10, 10))
    ndd = NDData(arr, wcs=image_2d_wcs)
    imviz_helper.load_data(ndd, data_label='dataset1')

    # force link to be pixel initially.
    link_image_data(imviz_helper.app, link_type='pixels')

    # apply elliptical subset
    theta = 45*u.deg.to(u.rad)
    imviz_helper.app.get_viewer('imviz-0').apply_roi(EllipticalROI(xc=5, yc=5,
                                                                   radius_x=2,
                                                                   radius_y=4,
                                                                   theta=theta))

    # get plugin and check that attribute tracking link type is set properly
    plugin = imviz_helper.plugins['Subset Tools']._obj
    assert plugin.display_sky_coordinates is False

    # get initial subset definitions from ROI applied
    subset_defs = plugin.subset_definitions

    # check that the subset definitions, which control what is displayed in the UI, are correct
    true_vals = (('X Center (pixels)', 5), ('Y Center (pixels)', 5),
                 ('X Radius (pixels)', 2), ('Y Radius (pixels)', 4),
                 ('Angle', theta*u.rad.to(u.deg)))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert subset_defs[0][i+1]['value'] == true_val[1]

    # get original subset location as a sky region for use later
    original_subs = imviz_helper.app.get_subsets(include_sky_region=True)
    original_sky_region = original_subs['Subset 1'][0]['sky_region']

    # move subset (subset state is what is modified in UI)
    plugin._set_value_in_subset_definition(0, 'X Center (pixels)', 'xc', 6)
    plugin._set_value_in_subset_definition(0, 'Y Center (pixels)', 'yc', 6)
    plugin._set_value_in_subset_definition(0, 'X Radius (pixels)', 'radius', 3)
    plugin._set_value_in_subset_definition(0, 'Y Radius (pixels)', 'radius', 5)
    plugin._set_value_in_subset_definition(0, 'Angle', 'radius', 0.)

    # update subset to apply these changes
    plugin.vue_update_subset()
    subset_defs = plugin.subset_definitions

    # and check that it is changed after vue_update_subset runs
    true_vals = (('X Center (pixels)', 6), ('Y Center (pixels)', 6),
                 ('X Radius (pixels)', 3), ('Y Radius (pixels)', 5),
                 ('Angle', 0.))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert subset_defs[0][i+1]['value'] == true_val[1]

    # get oupdated subset location as a sky region
    updated_sky_region = imviz_helper.app.get_subsets(include_sky_region=True)
    updated_sky_region = updated_sky_region['Subset 1'][0]['sky_region']

    # remove subsets and change link type to wcs
    dc = imviz_helper.app.data_collection
    dc.remove_subset_group(dc.subset_groups[0])
    link_image_data(imviz_helper.app, link_type='wcs')
    assert plugin.display_sky_coordinates is True  # linking change should trigger this to change to True

    # apply subset
    img_wcs = imviz_helper.app.data_collection[1].data.coords
    # make sure pixel positions are in the same place as first test case when pixel linked
    x, y = img_wcs.world_to_pixel(SkyCoord(original_sky_region.center.ra,
                                           original_sky_region.center.dec))
    imviz_helper.app.get_viewer('imviz-0').apply_roi(EllipticalROI(xc=x, yc=y,
                                                                   radius_x=2.,
                                                                   radius_y=4,
                                                                   theta=45*u.deg.to(u.rad)))

    # subset definition should now be in sky coordinates
    subset_defs = plugin.subset_definitions
    # check that the subset definitions, which control what is displayed in the UI, are correct
    true_vals = (('RA Center (degrees)', original_sky_region.center.ra.deg),
                 ('Dec Center (degrees)', original_sky_region.center.dec.deg),
                 ('RA Radius (degrees)', original_sky_region.height.to(u.deg).value / 2.),
                 ('Dec Radius (degrees)', original_sky_region.width.to(u.deg).value / 2.),
                 ('Angle', original_sky_region.angle.to(u.deg).value))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        assert_allclose(subset_defs[0][i+1]['value'], true_val[1])

    # move the to the same position as last time we updated it
    plugin._set_value_in_subset_definition(0, 'RA Center (degrees)', 'xc', updated_sky_region.center.ra.deg)
    plugin._set_value_in_subset_definition(0, 'Dec Center (degrees)', 'yc', updated_sky_region.center.dec.deg)
    plugin._set_value_in_subset_definition(0, 'RA Radius (degrees)', 'yc', updated_sky_region.height.to(u.deg).value / 2.)
    plugin._set_value_in_subset_definition(0, 'Dec Radius (degrees)', 'yc', updated_sky_region.width.to(u.deg).value / 2.)
    plugin._set_value_in_subset_definition(0, 'Angle', 'theta', updated_sky_region.angle.to(u.deg).value)

    # update subset
    plugin.vue_update_subset()

    subset_defs = plugin.subset_definitions
    true_vals = (('RA Center (degrees)', updated_sky_region.center.ra.deg),
                 ('Dec Center (degrees)', updated_sky_region.center.dec.deg),
                 ('RA Radius (degrees)', updated_sky_region.height.to(u.deg).value / 2.),
                 ('Dec Radius (degrees)', updated_sky_region.width.to(u.deg).value / 2.),
                 ('Angle', updated_sky_region.angle.to(u.deg).value))
    for i, true_val in enumerate(true_vals):
        assert subset_defs[0][i+1]['name'] == true_val[0]
        print(subset_defs[0][i+1]['value'], true_val[1])
        assert_allclose(subset_defs[0][i+1]['value'], true_val[1], atol=1e-6)