from pathlib import Path
import warnings
import pynapple as nap
from pynwb import NWBHDF5IO, NWBFile
from pynwb.behavior import SpatialSeries, Position, CompassDirection
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
import numpy as np
from pynwb.ecephys import ElectricalSeries
from pynwb.ecephys import LFP
from hdmf.backends.hdf5.h5_utils import H5DataIO
import scipy.io as spio


def create_nwb_file(metadata, start_time):
    # get info from folder name

    rec_id = metadata['file']['name'].split('-')
    print('Creating NWB file and adding metadata...')

    # create an nwb file
    nwbfile = NWBFile(
        session_description=metadata['file']['session_description'],
        experiment_description=metadata['file']['experiment_description'],
        identifier=rec_id[0],
        session_start_time=start_time,
        session_id=rec_id[1],
        experimenter=metadata['file']['experimenter'],
        lab=metadata['file']['lab'],
        institution=metadata['file']['institution'],
        virus='')

    # add subject
    age_weeks = metadata['subject']['age_weeks']
    nwbfile.subject = Subject(age=f'P{age_weeks}W',
                              description=metadata['subject']['line'],
                              species='Rattus norvegicus',
                              subject_id=rec_id[0],
                              genotype=metadata['subject']['genotype'],
                              sex=metadata['subject']['sex'])

    return nwbfile

def load_nwb_file(datapath, foldername):

    # load NWB file
    filepath = datapath / foldername
    filename = foldername + '.nwb'
    filepath = filepath / filename

    data = nap.load_file(str(filepath))
    print(data)

    return data

def save_nwb_file(nwbfile,datapath, foldername):
    print('Saving NWB file...')
    with NWBHDF5IO(datapath / foldername / (foldername + '.nwb'), 'w') as io:
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
            "events must be a dictionary where keys are labels and values are pynapple IntervalSet instances.")

    # Ensure all values in events are IntervalSet instances
    if not all(isinstance(interval_set, nap.IntervalSet) for interval_set in events.values()):
        raise TypeError("All values in events must be pynapple IntervalSet instances.")

    # Create a TimeIntervals table
    events_table = TimeIntervals(name=event_name)
    events_table.add_column(name="label", description="Type of event")

    for label, interval_set in events.items():
        for start, end in zip(interval_set['start'], interval_set['end']):
            events_table.add_row(start_time=start, stop_time=end, label=label)

    nwbfile.add_time_intervals(events_table)

    return nwbfile

def add_units(nwbfile, spikes, waveforms, shank_id):
    print('Adding units to NWB file...')

    shank_names = list(nwbfile.electrode_groups.keys())
    for ncell in range(len(spikes)):
        group_name = shank_names[shank_id[ncell]]  # Map shank_id to correct name
        nwbfile.add_unit(id=ncell,
                         spike_times=spikes[ncell].index.to_numpy(),
                         waveform_mean=waveforms[ncell].T,
                         electrode_group=nwbfile.electrode_groups[group_name])
    return nwbfile


def add_probes(nwbfile, metadata, xmldata):
    # to do: add depth info
    """
    Adds probes, electrode groups, and electrodes to the NWB file.
    Properly assigns shanks to probes when xmldata['spike_groups']
    is a sequential list instead of a dictionary.
    """
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
        for _ in range(len(electrodes)):
            nwbfile.add_electrode(
                group=electrode_group,
                location=electrode_group.location,
                filtering="none",
                imp=np.nan,  # Add real impedance values if available
            )

    # Define table region RAW DAT FILE and LFP will refer to (all electrodes)
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(electrodes))),
        description='all electrodes',
    )

    # Print how shanks are called
    print("Shank names:", shank_names)

    return nwbfile


def add_tracking(nwbfile, pos, ang=None):
    print('Adding tracking to NWB file...')

    # Create behavior module
    behavior_module = nwbfile.create_processing_module(
        name='behavior',
        description='Tracking data acquired with Bonsai'
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
    # EPOCHS

    print('Adding sleep stages...')

    sleep_file = sleep_path / (folder_name + '.SleepState.states.mat')
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

    return nwbfile


def add_epochs(nwbfile, epochs, metadata):
    print('Adding epochs to NWB file...')
    for epoch in range(epochs.shape[0]):
        nwbfile.add_epoch(start_time=float(epochs['Start'][epoch]), stop_time=float(epochs['End'][epoch]), tags=metadata['epoch'][str(epoch+1)])

    return nwbfile



def add_lfp(nwbfile, lfp_path, xml_data):

    print('Adding LFP...')

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(len(nwbfile.electrodes))),
        description='all electrodes',
    )

    # get channel numbers in shank order


    lfp_data = nap.load_eeg(filepath=lfp_path, channel=None, n_channels=xml_data['n_channels'], frequency=float(xml_data['eeg_sampling_rate']), precision='int16',
                            bytes_size=2)

    lfp_data = lfp_data[:, chanOrder]  # sort according to channel order

    # create ElectricalSeries
    lfp_elec_series = ElectricalSeries(
        name='LFP',
        data=H5DataIO(lfp_data, compression=True),  # use this function to compress
        description='Local field potential (low-pass filtered at 625 Hz)',
        electrodes=all_table_region,
        rate=1250.
    )

    # store ElectricalSeries in an LFP container
    warnings.filterwarnings("ignore",
                            message=".*DynamicTableRegion.*")  # this is to supress a warning here that doesn't seem cause any issues
    lfp = LFP(electrical_series=lfp_elec_series)
    warnings.resetwarnings()

    ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                      description='Processed electrophysiological signals'
                                                      )
    ecephys_module.add(lfp)

    return nwbfile





