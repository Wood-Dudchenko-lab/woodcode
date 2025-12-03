from pathlib import Path
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries, LFP, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection, BehavioralEvents
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common.table import DynamicTable
import scipy.io as spio
from datetime import datetime
import pprint
import numpy as np
import pandas as pd
import warnings
import pynapple as nap


def create_nwb_file(metadata, start_time):
    # get info from folder name

    rec_id = metadata['file']['name'].split('-')
    print('Creating NWB file and adding metadata...')

    # calculate animal age
    dob_str = str(metadata['subject']['dob'])
    dob = datetime(2000 + int(dob_str[:2]), int(dob_str[2:4]), int(dob_str[4:6]))
    age_days = (start_time.date() - dob.date()).days

    # create an nwb file
    nwbfile = NWBFile(
        session_description=metadata['file']['session_description'],
        experiment_description=metadata['file']['experiment_description'],
        identifier=metadata['file']['name'],
        session_start_time=start_time,
        session_id=rec_id[1],
        protocol=metadata['file']['protocol'],
        notes=metadata['file']['notes'],
        experimenter=metadata['file']['experimenter'],
        lab=metadata['file']['lab'],
        institution=metadata['file']['institution'],
        virus='')

    # add subject
    nwbfile.subject = Subject(age=f"P{age_days}D",
                              description=f"{metadata['subject']['line']} {int(metadata['subject']['stock_id'])}",
                              species='Rattus norvegicus',
                              subject_id=rec_id[0],
                              genotype=metadata['subject']['genotype'],
                              sex=metadata['subject']['sex'])

    return nwbfile


def load_nwb_file(file_path, file_name):
    # load NWB file
    file_name = file_name + '.nwb'
    data = nap.load_file(str(file_path / file_name))
    print(data)

    return data


def save_nwb_file(nwbfile, file_path, file_name):
    print('Saving NWB file...')

    # Create the folder path if it doesn't exist
    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)
        print(f'Created directory: {file_path}')

    # Save the NWB file
    with NWBHDF5IO(file_path / (file_name + '.nwb'), 'w') as io:
        io.write(nwbfile)

    print('Done!')


def add_events(nwbfile, events, event_name="events"):
    print('Adding events to NWB file...')

    # Handle case where events is a single IntervalSet
    if isinstance(events, nap.IntervalSet):
        events = {event_name: events}

    # Ensure events is a dictionary
    if not isinstance(events, dict):
        raise TypeError(
            "events must be a dictionary where keys are labels and values are pynapple IntervalSet instances."
        )

    # Ensure all values in events are IntervalSet instances
    if not all(isinstance(interval_set, nap.IntervalSet) for interval_set in events.values()):
        raise TypeError("All values in events must be pynapple IntervalSet instances.")

    # Convert events into a pandas DataFrame
    data = []
    for label, interval_set in events.items():
        df = pd.DataFrame({
            "start_time": interval_set["start"],
            "stop_time": interval_set["end"],
            "label": [label] * len(interval_set)
        })
        data.append(df)

    # Concatenate all event data
    events_df = pd.concat(data, ignore_index=True)

    # Create TimeIntervals from the DataFrame, now with a name
    events_table = TimeIntervals.from_dataframe(events_df, name=event_name)

    # Add to NWB file
    nwbfile.add_time_intervals(events_table)

    return nwbfile


def add_units(nwbfile, xml_data, spikes, waveforms, shank_id):
    print('Adding units to NWB file...')

    # Add extra unit column
    nwbfile.add_unit_column(name="sampling_rate", description="Sampling rate of the raw ephys signal")

    shank_names = list(nwbfile.electrode_groups.keys())
    for ncell in range(len(spikes)):
        group_name = shank_names[shank_id[ncell]]  # Map shank_id to correct name
        nwbfile.add_unit(id=ncell,
                         spike_times=spikes[ncell].index.to_numpy(),
                         waveform_mean=waveforms[ncell].T,
                         sampling_rate=xml_data['dat_sampling_rate'],
                         electrode_group=nwbfile.electrode_groups[group_name])
    return nwbfile


def add_probes(nwbfile, metadata, xmldata, nrsdata):
    # to do: add depth info
    """
    Adds probes, electrode groups, and electrodes to the NWB file.
    Properly assigns shanks to probes when xmldata['spike_groups']
    is a sequential list instead of a dictionary.
    """

    # get a list of dead channels from the nrs file
    good_channels = nrsdata['channels_shown']

    # Add extra electrode columns
    nwbfile.add_electrode_column(name='label', description='label of electrode')
    nwbfile.add_electrode_column(name='is_faulty', description='Boolean column to indicate faulty electrodes')

    # Add probes as devices
    probe_devices = {}
    for probe in metadata["probe"]:
        probe_name = f"Probe {probe['id']}"
        probe_devices[probe['id']] = nwbfile.create_device(
            name=probe_name,
            description=probe["description"],
            manufacturer=probe.get("type", "Unknown Manufacturer"),
        )

    # Determine how many shanks belong to each probe
    shank_assignments = []
    for probe in metadata["probe"]:
        probe_id = probe["id"]
        nshanks = probe["nshanks"]
        shank_assignments.extend(
            [(probe_id, shank_num + 1, probe["location"], probe["step"]) for shank_num in range(nshanks)])

    # Ensure number of shanks in metadata matches xmldata
    if len(shank_assignments) != len(xmldata["spike_groups"]):
        raise ValueError("Mismatch between shank count in metadata and xmldata['spike_groups']")

    # Add electrode groups and electrodes
    electrode_counter = 0
    shank_names = []
    for (probe_id, shank_id, probe_location, probe_step), (shank_idx, electrodes) in zip(shank_assignments, enumerate(
            xmldata["spike_groups"])):

        # Create Electrode Group
        group_name = f"probe{probe_id}shank{shank_id}"
        shank_names.append(group_name)
        electrode_group = nwbfile.create_electrode_group(
            name=group_name,
            description=f"Electrodes from {group_name}, step: {probe_step}",
            location=probe_location,
            device=probe_devices[probe_id],
        )

        # Add electrodes to the NWB electrode table
        for ielec in range(len(electrodes)):
            elec_depth = probe_step * (len(electrodes) - ielec - 1)
            elec_label = f"{group_name}elec{ielec}"
            nwbfile.add_electrode(
                x=0., y=float(elec_depth), z=0.,  # add electrode position
                group=electrode_group,
                is_faulty=electrode_counter not in good_channels,
                location=electrode_group.location,
                filtering="none",
                label=elec_label,
                imp=np.nan,  # Add real impedance values if available
            )
            electrode_counter += 1

    # Define table region RAW DAT FILE and LFP will refer to (all electrodes)
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(electrodes))),
        description='all electrodes',
    )

    # Print how shanks are called
    # print("Shank names:", shank_names)

    return nwbfile


def add_tracking(nwbfile, pos, ang=None):
    # to do: add units as input
    print('Adding tracking to NWB file...')

    # Create behavior module
    behavior_module = nwbfile.create_processing_module(
        name='behavior',
        description='Behavioral data'
    )

    # Create the spatial series for position
    spatial_series_obj = SpatialSeries(
        name='position',
        description='(x,y) position',
        data=pos.values,
        timestamps=pos.index.to_numpy(),
        reference_frame='',
        unit='centimeters'
    )
    position_obj = Position(spatial_series=spatial_series_obj)
    behavior_module.add(position_obj)

    # Add head-direction data only if ang is provided
    if ang is not None:
        spatial_series_obj = SpatialSeries(
            name='head-direction',
            description='Horizontal angle of the head (yaw)',
            data=ang.values,
            timestamps=ang.index.to_numpy(),
            reference_frame='',
            unit='radians'
        )
        direction_obj = CompassDirection(spatial_series=spatial_series_obj)
        behavior_module.add(direction_obj)

    return nwbfile


def add_sleep(nwbfile, sleep_path, folder_name):
    sleep_file = sleep_path / (folder_name + '.SleepState.states.mat')
    emg_file = sleep_path / (folder_name + '.EMGFromLFP.LFP.mat')

    print('Adding sleep stages to NWB file...')

    sleepEpochs = spio.loadmat(sleep_file, simplify_cells=True)
    epWake = np.float32(sleepEpochs['SleepState']['ints']['WAKEstate'])
    epNREM = np.float32(sleepEpochs['SleepState']['ints']['NREMstate'])
    epREM = np.float32(sleepEpochs['SleepState']['ints']['REMstate'])

    sleep_stages = TimeIntervals(name='sleep_stages')

    if epREM.size > 0:
        if epREM.ndim == 1:  # in case there is only one interval
            sleep_stages.add_row(start_time=epREM[0], stop_time=epREM[1], tags=['rem'])  # tags need to go as list
        elif epREM.ndim == 2:
            for nrow in range(len(epREM)):
                sleep_stages.add_row(start_time=epREM[nrow, 0], stop_time=epREM[nrow, 1], tags=['rem'])

    if epNREM.size > 0:
        if epNREM.ndim == 1:  # in case there is only one interval
            sleep_stages.add_row(start_time=epNREM[0], stop_time=epNREM[1], tags=['nrem'])
        elif epNREM.ndim == 2:
            for nrow in range(len(epNREM)):
                sleep_stages.add_row(start_time=epNREM[nrow, 0], stop_time=epNREM[nrow, 1], tags=['nrem'])

    if epWake.size > 0:
        if epWake.ndim == 1:  # in case there is only one interval
            sleep_stages.add_row(start_time=epWake[0], stop_time=epWake[1], tags=['wake'])
        elif epWake.ndim == 2:
            for nrow in range(len(epWake)):
                sleep_stages.add_row(start_time=epWake[nrow, 0], stop_time=epWake[nrow, 1], tags=['wake'])

    nwbfile.add_time_intervals(sleep_stages)

    print('Adding pseudoEMG to the NWB file...')

    emg = spio.loadmat(emg_file, simplify_cells=True)
    emg = TimeSeries(
        name="pseudoEMG",
        description="Pseudo EMG from correlated high-frequency LFP",
        data=emg['EMGFromLFP']['data'],
        unit="a.u.",
        timestamps=emg['EMGFromLFP']['timestamps']
    )

    # Create an extracellular ephys module or add to the existing one
    if 'ecephys' not in nwbfile.processing:
        ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                          description='Processed electrophysiological signals'
                                                          )
        ecephys_module.add(emg)
    else:
        nwbfile.processing['ecephys'].add(emg)

    return nwbfile


def add_epochs(nwbfile, epochs, metadata):
    """
    Adds epochs to an NWB file.

    Parameters:
    - nwbfile: NWBFile object
    - epochs: DataFrame containing 'Start' and 'End' times
    - metadata: Dictionary containing epoch metadata

    Returns:
    - Updated NWBFile object
    """

    print('Adding epochs to NWB file...')

    # Extract all tags from metadata
    epoch_tags = {str(i + 1): metadata['epoch'].get(str(i + 1), None) for i in range(epochs.shape[0])}

    # Check that all epochs have corresponding tags
    if None in epoch_tags.values():
        missing_epochs = [k for k, v in epoch_tags.items() if v is None]
        raise ValueError(f"Missing tags for epochs: {missing_epochs}")

    # Add epochs to NWB file
    for epoch in range(epochs.shape[0]):
        nwbfile.add_epoch(
            start_time=float(epochs['Start'][epoch]),
            stop_time=float(epochs['End'][epoch]),
            tags=epoch_tags[str(epoch + 1)]
        )

    return nwbfile


def add_lfp(nwbfile, lfp_path, xml_data):
    print('Adding LFP to the NWB file...')

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(nwbfile.electrodes))),
        description='all electrodes',
    )

    # get channel numbers in shank order
    chan_order = np.concatenate(xml_data['spike_groups'])

    # lazy load LFP
    lfp_data = nap.load_eeg(filepath=lfp_path, channel=None, n_channels=xml_data['n_channels'],
                            frequency=float(xml_data['eeg_sampling_rate']), precision='int16',
                            bytes_size=2)
    lfp_data = lfp_data[:, chan_order]  # get only probe channels

    # create ElectricalSeries
    lfp_elec_series = ElectricalSeries(
        name='LFP',
        data=H5DataIO(lfp_data, compression=True),  # use this function to compress
        description='Local field potential (downsampled DAT file)',
        electrodes=all_table_region,
        rate=float(xml_data['eeg_sampling_rate'])
    )

    # store ElectricalSeries in an LFP container
    warnings.filterwarnings("ignore",
                            message=".*DynamicTableRegion.*")  # this is to supress a warning here that doesn't seem cause any issues
    lfp = LFP(electrical_series=lfp_elec_series)
    warnings.resetwarnings()

    # Create an extracellular ephys module or add to the existing one
    if 'ecephys' not in nwbfile.processing:
        ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                          description='Processed electrophysiological signals'
                                                          )
        ecephys_module.add(lfp)
    else:
        nwbfile.processing['ecephys'].add(lfp)

    return nwbfile


def add_misc_tsd(nwbfile, tsd, name='unnamed_series', description='', unit=''):
    """
    Adds a pynapple Tsd or TsdFrame to the 'misc' processing module in an NWB file.

    Parameters:
    - nwbfile: NWBFile object
    - tsd: pynapple Tsd or TsdFrame
    - name: Name of the TimeSeries
    - description: Description of the TimeSeries
    - unit: Unit of the TimeSeries data

    Returns:
    - Updated NWBFile object
    """
    print(f"Adding '{name}' to the NWB file...")

    # grab or create the 'misc' module
    if 'misc' not in nwbfile.processing:
        module = nwbfile.create_processing_module(
            name='misc',
            description='Miscellaneous data'
        )
    else:
        module = nwbfile.processing['misc']

    # make a TimeSeries

    misc_series = TimeSeries(
        name=name,
        data=tsd.values,
        unit=unit,
        timestamps=tsd.index.values
    )

    module.add(misc_series)
    return nwbfile


def collect_nwb_metadata(nwbfile):
    """
    Collects and displays all metadata from an NWB file in a structured format.

    Parameters:
    - nwbfile: An NWBFile object.

    Returns:
    - A dictionary containing all extracted metadata.
    """

    metadata = {}

    # Extract top-level metadata
    metadata["General Info"] = {
        "session_description": nwbfile.session_description,
        "identifier": nwbfile.identifier,
        "session_start_time": nwbfile.session_start_time,
        "experimenter": nwbfile.experimenter,
        "lab": nwbfile.lab,
        "institution": nwbfile.institution,
        "experiment_description": nwbfile.experiment_description,
        "related_publications": nwbfile.related_publications,
        "keywords": nwbfile.keywords,
        "notes": nwbfile.notes,
        "pharmacology": nwbfile.pharmacology,
        "protocol": nwbfile.protocol,
        "slices": nwbfile.slices,
        "source_script": nwbfile.source_script,
        "source_script_file_name": nwbfile.source_script_file_name,
        "data_collection": nwbfile.data_collection,
    }

    # Extract subject metadata
    if nwbfile.subject:
        metadata["Subject"] = {
            "subject_id": nwbfile.subject.subject_id,
            "age": nwbfile.subject.age,
            "description": nwbfile.subject.description,
            "species": nwbfile.subject.species,
            "genotype": nwbfile.subject.genotype,
            "sex": nwbfile.subject.sex,
            "weight": nwbfile.subject.weight,
            "date_of_birth": nwbfile.subject.date_of_birth,
            "strain": getattr(nwbfile.subject, "strain", "N/A"),  # Optional field
        }

    # Extract device metadata
    metadata["Devices"] = {name: device.description for name, device in nwbfile.devices.items()}

    # Extract electrode group metadata
    metadata["Electrode Groups"] = {
        name: {
            "description": group.description,
            "location": group.location,
            "device": group.device.name
        }
        for name, group in nwbfile.electrode_groups.items()
    }

    # Extract electrode table metadata
    if nwbfile.electrodes is not None:
        metadata["Electrodes"] = nwbfile.electrodes.to_dataframe().to_dict()

    # Extract acquisition metadata
    metadata["Acquisition"] = {name: type(obj).__name__ for name, obj in nwbfile.acquisition.items()}

    # Extract processing modules
    metadata["Processing Modules"] = {}
    for module_name, module in nwbfile.processing.items():
        module_data = {}
        for data_name, data in module.data_interfaces.items():
            module_data[data_name] = type(data).__name__
        metadata["Processing Modules"][module_name] = module_data

    # Extract intervals metadata (e.g., epochs)
    metadata["Time Intervals"] = {}
    for interval_name, interval_table in nwbfile.intervals.items():
        if isinstance(interval_table, DynamicTable):
            metadata["Time Intervals"][interval_name] = interval_table.to_dataframe().to_dict()

    # Extract analysis metadata
    metadata["Analysis"] = {name: type(obj).__name__ for name, obj in
                            nwbfile.analysis.items()} if nwbfile.analysis else {}

    # Extract lab metadata
    metadata["Lab Metadata"] = {name: type(obj).__name__ for name, obj in
                                nwbfile.lab_meta_data.items()} if nwbfile.lab_meta_data else {}

    # Print metadata in a structured way
    print("\n=== NWB Metadata ===")
    pprint.pprint(metadata, width=120, compact=False)

    return metadata


def create_spatial_series(name, data, description, unit):
    """
    Create a SpatialSeries object for behavioral data.

    Parameters
    ----------
    name : str
        Name of the spatial series
    data : TsdFrame or Tsd
        Behavioral data
    description : str, optional
        Description of the data
    unit : str, optional
        Unit of measurement

    Returns
    -------
    spatial_series_obj
        The created SpatialSeries object
    """
    spatial_series_obj = SpatialSeries(
        name=name,
        description=description,
        data=data.values,
        timestamps=data.index.to_numpy(),
        reference_frame='',
        unit=unit
    )
    return spatial_series_obj


def add_behavioural_data(nwbfile, data, name, description, unit):
    """
    Add behavioral data to nwbfile.

    Parameters
    ----------
    nwbfile: NWBFile
    data : TsdFrame or Tsd
        Behavioral data
    name : str
        Name of the spatial series
    description : str, optional
        Description of the data
    unit : str, optional
        Unit of measurement

    Returns
    -------
    nwbfile
        nwbfile with behavioral data added to it
    """
    if 'behavior' in nwbfile.processing:
        behavior_module = nwbfile.processing['behavior']
    else:
        behavior_module = nwbfile.create_processing_module(
            name='behavior',
            description='Behavioral data'
        )
    spatial_series_obj = create_spatial_series(name, data, description, unit)
    position_obj = Position(spatial_series=spatial_series_obj)
    behavior_module.add(position_obj)

    return nwbfile


def add_single_events(nwbfile, data, name, description, unit):
    """
    Add behavioral events to nwbfile.

    Parameters
    ----------
    nwbfile: NWBFile
    data : Tsd
        Behavioral event data
    name : str
        Name of the time series
    description : str, optional
        Description of the data
    unit : str, optional
        Unit of measurement

    Returns
    -------
    nwbfile
        nwbfile with behavioral data added to it
    """
    if 'behavior' in nwbfile.processing:
        behavior_module = nwbfile.processing['behavior']
    else:
        behavior_module = nwbfile.create_processing_module(
            name='behavior',
            description='Behavioral data'
        )
    time_series = TimeSeries(
        name=name,
        data=data.values,
        timestamps=data.index.to_numpy(),
        description=description,
        unit=unit,
    )
    behavioral_events = BehavioralEvents(time_series=time_series, name=name)
    behavior_module.add(behavioral_events)

    return nwbfile

