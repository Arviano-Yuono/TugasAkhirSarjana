import vtk
import h5py
import numpy as np

# Step 1: Read the .vtr file
filename = 'Data/field000004.vtr' # Path to your .vtr file

reader = vtk.vtkXMLRectilinearGridReader()
reader.SetFileName(filename)
reader.Update()

# Get the output data
rectilinear_grid = reader.GetOutput()

# Step 2: Extract data arrays
point_data = rectilinear_grid.GetPointData()  # Data associated with points
cell_data = rectilinear_grid.GetCellData()    # Data associated with cells


# Extract coordinates
x = np.array(rectilinear_grid.GetXCoordinates())
y = np.array(rectilinear_grid.GetYCoordinates())
z = np.array(rectilinear_grid.GetZCoordinates())

# Create a dictionary to store all extracted data
data_dict = {
    "coordinates": {"x": x, "y": y, "z": z},
    "point_data": {},
    "cell_data": {},
}

# Extract scalar fields from point data
for i in range(point_data.GetNumberOfArrays()):
    array_name = point_data.GetArrayName(i)
    array_data = vtk.util.numpy_support.vtk_to_numpy(point_data.GetArray(i))
    data_dict["point_data"][array_name] = array_data

# Extract scalar fields from cell data
for i in range(cell_data.GetNumberOfArrays()):
    array_name = cell_data.GetArrayName(i)
    array_data = vtk.util.numpy_support.vtk_to_numpy(cell_data.GetArray(i))
    data_dict["cell_data"][array_name] = array_data

# Step 3: Save to .h5 format
h5_file_path = "output_data.h5"  # Specify the output .h5 file name
with h5py.File(h5_file_path, "w") as h5_file:
    # Save coordinates
    coord_group = h5_file.create_group("coordinates")
    for axis, coords in data_dict["coordinates"].items():

        coord_group.create_dataset(axis, data=coords)

    # Save point data
    point_group = h5_file.create_group("point_data")
    for name, array in data_dict["point_data"].items():
        point_group.create_dataset(name, data=array)

    # Save cell data
    cell_group = h5_file.create_group("cell_data")
    for name, array in data_dict["cell_data"].items():
        cell_group.create_dataset(name, data=array)

print(f"Data successfully converted to {h5_file_path}")
