
from astropy.table import QTable, Table
from jdaviz.core.loaders.importers.catalog.catalog import CatalogImporter
import pytest


def test_coord_column_detection(deconfigged_helper):
    """
    Test automatic detection of RA/Dec columns with various naming
    conventions, and that non-coordinate columns are not misidentified as
    coordinate columns (e.g 'radial' which contains 'ra' but should not be
    identified as 'Ra').
    """

    # Variations of 'ra' and 'dec' that should be correctly identified as coordinate columns
    ra_variations = ['rightascension', 'ra', 'radeg',
                     'radegrees', 'rightascensiondegrees', 'rightascensiondeg',
                     'raobj', 'objra', 'sourcera', 'rasource', 'raj2000', 'ra2000',
                     'worldra', 'targra', 'scira']
    dec_variations = ['declination', 'dec', 'decdeg',
                      'decdegrees', 'declinationdegrees', 'declinationdeg',
                      'decobj', 'objdec', 'decsource', 'sourcedec', 'decj2000',
                      'dec2000', 'worlddec', 'targdec', 'scidec']

    variations_to_pass = list(zip(ra_variations, dec_variations))

    for v in variations_to_pass:
        ra, dec = v  # unpack RA and Dec column names
        tab = QTable({ra: [10.0], dec: [-5.0]})

        ldr = deconfigged_helper.loaders['object']
        ldr.object = tab
        ldr.format = 'Catalog'
        importer = ldr.importer

        # make sure the coordinate columns were correctly identified
        assert importer.col_ra == ra
        assert importer.col_dec == dec

    # check that certain strings that contain 'ra' and 'dec' substrings are not
    # misidentified as coordinate columns
    tab = QTable({'radial_velocity': [10.0], 'fluxradius': [5.0], 'decrement': [1.0]})
    ldr = deconfigged_helper.loaders['object']
    ldr.object = tab
    ldr.format = 'Catalog'
    importer = ldr.importer
    # none of the column names in the input table should have been identified as RA or Dec columns,
    # so they should be set as a placeholder value of '---'
    assert importer.col_ra == '---'
    assert importer.col_dec == '---'


@pytest.mark.parametrize("pixel_name", ['x', 'y'])
def test_pixel_column(deconfigged_helper,
                      sky_coord_only_source_catalog,
                      pixel_name):

    resolver = deconfigged_helper.loaders['object']._obj
    importer = CatalogImporter(app=deconfigged_helper._app,
                               resolver=resolver, parser=None,
                               input=sky_coord_only_source_catalog)

    sky_coord_only_source_catalog.rename_column('ra', 'x')
    sky_coord_only_source_catalog.rename_column('dec', 'y')

    variations_to_check = {(pixel_name.upper(), pixel_name.upper()),
                           (pixel_name + 'source', pixel_name + 'source'),
                           (pixel_name + '_pix', pixel_name + '_pix'),
                           ('pix_' + pixel_name, 'pix_' + pixel_name),
                           ('galaxy', '---')
                           }

    for v in variations_to_check:
        this_table = sky_coord_only_source_catalog.copy()
        this_table.rename_column(pixel_name, v[0])
        importer._input = this_table
        assert importer._guess_coord_cols(pixel_name)[0] == v[1]


def test_catalog_importer_is_valid(deconfigged_helper):
    """Test _check_is_valid for CatalogImporter: success and failure cases."""
    resolver = deconfigged_helper.loaders['object']._obj

    # Success: non-empty table
    importer = CatalogImporter(app=deconfigged_helper._app,
                               resolver=resolver, parser=None,
                               input=Table({'ra': [10.0, 20.0], 'dec': [-5.0, 10.0]}))
    assert importer._check_is_valid() == ''

    # Failure: non-table input
    importer._input = 'not_a_catalog'
    assert importer._check_is_valid() == 'Input is not a valid catalog.'

    # Failure: empty table
    importer._input = Table()
    assert importer._check_is_valid() == 'Input is not a valid catalog.'
