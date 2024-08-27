from astropy import units as u
import itertools

__all__ = ['units_to_strings', 'create_spectral_equivalencies_list',
           'create_flux_equivalencies_list', 'check_if_unit_is_per_solid_angle']


def units_to_strings(unit_list):
    """Convert equivalencies into readable versions of the units.

    Parameters
    ----------
    unit_list : list
        List of either `astropy.units` or strings that can be converted
        to `astropy.units`.

    Returns
    -------
    result : list
        A list of the units with their best (i.e., most readable) string version.
    """
    return [u.Unit(unit).to_string() for unit in unit_list]


def create_spectral_equivalencies_list(spectral_axis_unit,
                                       exclude=[u.jupiterRad, u.earthRad, u.solRad,
                                                u.lyr, u.AU, u.pc, u.Bq, u.micron, u.lsec]):
    """Get all possible conversions from current spectral_axis_unit."""
    if spectral_axis_unit in (u.pix, u.dimensionless_unscaled):
        return []

    # Get unit equivalencies.
    try:
        curr_spectral_axis_unit_equivalencies = spectral_axis_unit.find_equivalent_units(
            equivalencies=u.spectral())
    except u.core.UnitConversionError:
        return []

    # Get local units.
    locally_defined_spectral_axis_units = ['Angstrom', 'nm',
                                           'um', 'Hz', 'erg']
    local_units = [u.Unit(unit) for unit in locally_defined_spectral_axis_units]

    # Remove overlap units.
    curr_spectral_axis_unit_equivalencies = list(set(curr_spectral_axis_unit_equivalencies)
                                                 - set(local_units + exclude))

    # Convert equivalencies into readable versions of the units and sorted alphabetically.
    spectral_axis_unit_equivalencies_titles = sorted(units_to_strings(
        curr_spectral_axis_unit_equivalencies))

    # Concatenate both lists with the local units coming first.
    return sorted(units_to_strings(local_units)) + spectral_axis_unit_equivalencies_titles


def create_flux_equivalencies_list(flux_unit, spectral_axis_unit):
    """Get all possible conversions for flux from current flux units."""
    if ((flux_unit in (u.count, u.dimensionless_unscaled))
            or (spectral_axis_unit in (u.pix, u.dimensionless_unscaled))):
        return []

    # Get unit equivalencies. Value passed into u.spectral_density() is irrelevant.
    try:
        curr_flux_unit_equivalencies = flux_unit.find_equivalent_units(
            equivalencies=u.spectral_density(1 * spectral_axis_unit),
            include_prefix_units=False)
    except u.core.UnitConversionError:
        return []

    mag_units = ['bol', 'AB', 'ST']
    # remove magnitude units from list
    curr_flux_unit_equivalencies = [unit for unit in curr_flux_unit_equivalencies if not any(mag in unit.name for mag in mag_units)]  # noqa

    # Get local flux units.
    locally_defined_flux_units = ['Jy', 'mJy', 'uJy', 'MJy',
                                  'W / (Hz m2)',
                                  'eV / (s m2 Hz)',
                                  'erg / (s cm2 Hz)',
                                  'erg / (s cm2 Angstrom)',
                                  'ph / (Angstrom s cm2)',
                                  'ph / (Hz s cm2)',
                                  ]
    local_units = [u.Unit(unit) for unit in locally_defined_flux_units]

    # Remove overlap units.
    curr_flux_unit_equivalencies = list(set(curr_flux_unit_equivalencies)
                                        - set(local_units))

    # Convert equivalencies into readable versions of the units and sort them alphabetically.
    flux_unit_equivalencies_titles = sorted(units_to_strings(curr_flux_unit_equivalencies))

    # Concatenate both lists with the local units coming first.
    return sorted(units_to_strings(local_units)) + flux_unit_equivalencies_titles


def create_angle_equivalencies_list(unit):
    # first, convert string to u.Unit obj.
    # this will take care of some formatting consistency like
    # turning something like Jy / (degree*degree) to Jy / deg**2
    # and erg sr^1 to erg / sr
    if isinstance(unit, u.core.Unit) or isinstance(unit, u.core.CompositeUnit):
        unit_str = unit.to_string()
    elif isinstance(unit, str):
        unit = u.Unit(unit)
        unit_str = unit.to_string()
    elif unit == 'ct':
        return ['pix^2']
    else:
        raise ValueError('Unit must be u.Unit, or string that can be converted into a u.Unit')

    if '/' in unit_str:
        # might be comprised of several units in denom.
        denom = unit_str.split('/')[-1].split()
        return denom
    else:
        # this could be where force / u.pix
        return ['pix^2']

def new_create_angle_equivalencies_list(solid_angle_unit):

    # replace create_angle_equivalencies_list with this

    # cast to unit
    # this can be removed when we make sure every time this is called
    # solid_angle_unit is already a string
    if isinstance(solid_angle_unit, str):
        solid_angle_unit = u.Unit(solid_angle_unit)

    # if data is flux unit or counts, represent SB as per pixel
    if solid_angle_unit is None:
        equivs = ['pix']
    # otherwise, we can convert between input solid angle unit (e.g deg**2),
    # steradians, and pixels. This is where we can add more options in the future.
    else:
        unit_str = solid_angle_unit.to_string()
        equivs = ['sr', 'pix']
        if unit_str not in equivs:
            equivs  = [unit_str] + equivs

    return equivs


def check_if_unit_is_per_solid_angle(unit, return_unit=False):
    """
    Check if a given u.Unit or unit string (that can be converted to
    a u.Unit object) represents some unit per solid angle. If 'return_unit'
    is True, then the 

    Parameters
    ----------
    unit : str or u.Unit
        Unit object or string representation of unit.
    return_unit : bool
        If True, the u.Unit of the solid angle unit will
        be returned (or None if unit is not a solid angle).

    Examples
    --------
    >>> check_if_unit_is_per_solid_angle('erg / (s cm^2 sr)')
    True
    >>> check_if_unit_is_per_solid_angle('erg / (s cm^2 sr)', return_unit=True)
    u.sr
    >>> check_if_unit_is_per_solid_angle('erg / s cm^2')
    False
    >>> check_if_unit_is_per_solid_angle('erg / s cm^2', return_unit=True)
    None
    >>> check_if_unit_is_per_solid_angle('Jy * sr^-1')
    True

    """

    # first, convert string to u.Unit obj.
    # this will take care of some formatting consistency like
    # turning something like Jy / (degree*degree) to Jy / deg**2
    # and erg sr^1 to erg / sr
    if isinstance(unit, u.core.Unit) or isinstance(unit, u.core.CompositeUnit):
        unit_str = unit.to_string()
    elif isinstance(unit, u.core.IrreducibleUnit):  # u.count
        unit_str = unit.to_string()
    elif isinstance(unit, str):
        unit = u.Unit(unit)
        unit_str = unit.to_string()
    else:
        raise ValueError('Unit must be u.Unit, or string that can be converted into a u.Unit')

    if '/' in unit_str:
        # input unit might be comprised of several units in denom. so check all.
        denom = unit_str.split('/')[-1].split()

        # find all combos of one or two units, to catch cases where there are
        # two different units of angle in the denom that might comprise a solid
        # angle when multiplied.
        for i in [combo for length in (1, 2) for combo in itertools.combinations(denom, length)]:
            # turn tuple of 1 or 2 units into a string, and turn that into a u.Unit
            # to check type
            new_unit_str = ' '.join(i).translate(str.maketrans('', '', '()'))
            new_unit = u.Unit(new_unit_str)
            if new_unit.physical_type == 'solid angle':
                if return_unit:  # area units present and requested to be returned
                    return new_unit
                return True  # area units present but not requested to be returned
            # square pixel should be considered a square angle unit
            if new_unit == u.pix * u.pix:
                if return_unit:
                    return new_unit
                return True

    # in the case there are no area units, but return units were requested
    if return_unit:
        return None

    # and if there are no area units, and return units were NOT requested.
    return False

