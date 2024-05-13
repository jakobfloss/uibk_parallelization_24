import h5py
from enum import Enum
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


path_data = "/put/your/path/here/code/build"


class plane_type(Enum) :
    xy = 1
    xz = 2
    yz = 3

class data_reader :
    def __init__(self, file_name) :
        self.file_name = file_name
        self.open_file(file_name)
        
        
    def open_file(self,file_name) :
        self.h5_file = h5py.File(file_name)
        self.h5group = self.h5_file['Data']
        
    def get_x_grid_left(self) :
        return self.x_grid_left
    
    def get_y_grid_left(self) :
        return self.y_grid_left
    
    def get_x_grid_cen(self) :
        return self.x_grid_cen
    
    def get_y_grid_cen(self) :
        return self.y_grid_cen
    
    def get_grid_extent(self) :
        x_grid_cen = self.h5group["x_grid_centers"]
        y_grid_cen = self.h5group["y_grid_centers"]
        z_grid_cen = self.h5group["z_grid_centers"]
        num_x = x_grid_cen.size
        num_y = y_grid_cen.size
        num_z = z_grid_cen.size
        return num_x,num_y,num_z
        
    def get_data_plane(self,name_dataset, plane_select, index_perp) :
        # get full 3D dataset
        raw_dataset = self.h5group[name_dataset]
        x_grid_left_global = self.h5group["x_grid_left"]
        y_grid_left_global = self.h5group["y_grid_left"]
        z_grid_left_global = self.h5group["z_grid_left"]
        
        x_grid_cen_global = self.h5group["x_grid_centers"]
        y_grid_cen_global = self.h5group["y_grid_centers"]
        z_grid_cen_global = self.h5group["z_grid_centers"]
        
        if(plane_select==plane_type.xy) :
            plane_data = raw_dataset[:,:,index_perp]
            self.x_grid_left = x_grid_left_global
            self.y_grid_left = y_grid_left_global
            self.x_grid_cen = x_grid_cen_global
            self.y_grid_cen = y_grid_cen_global
        elif(plane_select==plane_type.xz) :
            plane_data = raw_dataset[:,index_perp,:]
            self.x_grid_left = x_grid_left_global
            self.y_grid_left = z_grid_left_global
            self.x_grid_cen = x_grid_cen_global
            self.y_grid_cen = z_grid_cen_global
        else :
            plane_data = raw_dataset[index_perp,:,:]
            self.x_grid_left = y_grid_left_global
            self.y_grid_left = z_grid_left_global
            self.x_grid_cen = y_grid_cen_global
            self.y_grid_cen = z_grid_cen_global
        return plane_data


class data_plotter() :
    def __init__(self) :
        self.color_map = plt.cm.coolwarm
    
    def plot(data, x_left, y_left, output_name) :
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        plt.pcolormesh(x_left, y_left, data)
        
        ax.set_aspect('equal')
            
        fig.savefig(output_name)
        
    def plot_3D(data_xy, x_grid, y_grid, output_name) :
        
        #fig = plt.figure()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #ax = plt.figure().add_subplot(111, projection='3d')
        
        cmap = plt.cm.plasma
        
        x_plot, y_plot = np.meshgrid(x_grid, y_grid)
        
        ax.plot_surface(x_plot, y_plot, data_xy, vmin=data_xy.min() * 2,
                        rstride=1, cstride=1, cmap=cmap)
        ax.azim = 30
        ax.elev = 20
        #ax.plot_surface(data_xy, x_left, y_left,
        #                rstride=1, cstride=1, facecolors=cmap(data_xy), shade=False)
        
        fig.savefig(output_name)
        
        
    def plot_3D_projected(data_xy, data_xz, data_yz, x_grid, y_grid, z_grid,
                          x_left, y_left, z_left, output_name) :
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cmap = plt.cm.plasma
        
        
        
        x_plot_xy, y_plot_xy = np.meshgrid(x_grid, y_grid)
        x_plot_xz, z_plot_xz = np.meshgrid(x_grid, z_grid)
        y_plot_yz, z_plot_yz = np.meshgrid(y_grid, z_grid)
        
        #ax.plot_surface(x_plot, y_plot, data_xy, rstride=8, cstride=8, alpha=0.3)
        x_min = x_left[0]
        y_min = y_left[0]
        z_min = z_left[0]
        
        #ax = plt.axes(projection='3d')
        
        
        cset = ax.contourf(x_plot_xy, y_plot_xy, data_xy,
                           zdir='z', offset=x_min, cmap=plt.cm.coolwarm,zorder=-1)
        cset = ax.contourf(x_plot_xz, data_xz, z_plot_xz, zdir='y',
                           offset=y_min, cmap=plt.cm.coolwarm,zorder=2)
        cset = ax.contourf(data_yz, y_plot_yz, z_plot_yz, zdir='x',
                           offset=z_min, cmap=plt.cm.coolwarm,zorder=3)
        #cset = ax.contour(x_plot_xy, y_plot_xy, data_xy, zdir='z', offset=z_min, cmap=plt.cm.coolwarm)
        
        #ax.plot_surface(x_plot, y_plot, data_xy, vmin=data_xy.min() * 2,
        #                rstride=1, cstride=1, cmap=cmap)
        ax.azim = 30
        ax.elev = 20
        
        ax.set_aspect('equal')
        ax.set_box_aspect([1,1,1])
        
        ax.set_xlim([np.min(x_grid), np.max(x_grid)])
        ax.set_ylim([np.min(y_grid), np.max(y_grid)])
        ax.set_zlim([np.min(z_grid), np.max(z_grid)])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #ax.set_zlim([-0.5,2])
        
        #ax.plot_surface(data_xy, x_left, y_left,
        #                rstride=1, cstride=1, facecolors=cmap(data_xy), shade=False)
        
        fig.savefig(output_name)

def analysis_single_plane() :
    file_name = "output_step140.h5"
    full_name = path_data + "/" + file_name;
    my_reader = data_reader(full_name)
    
    size_x, size_y, size_z = my_reader.get_grid_extent()
    print("Size of grid",size_x, size_y, size_z)
    
    i_mid_x = int(size_x/2)
    i_mid_y = int(size_y/2)
    i_mid_z = int(size_z/2)
    
    data_2D = my_reader.get_data_plane("Density",plane_type.yz, i_mid_x)
    x_grid = my_reader.get_x_grid_left()
    y_grid = my_reader.get_y_grid_left()
    print(y_grid[:])
    print("data near middle:",data_2D[i_mid_y-1,i_mid_z], data_2D[i_mid_y,i_mid_z])
    
    plotter = data_plotter
    plotter.plot(data_2D, x_grid, y_grid, "test.pdf")

def analysis_3D() :
    # Here we do a joint plot of all 3 midplanes
    file_name = "output_step140.h5"
    full_name = path_data + "/" + file_name;
    my_reader = data_reader(full_name)    
    
    # Now, get the data in the 3 mid planes
    size_x, size_y, size_z = my_reader.get_grid_extent()
    print("Size of grid",size_x, size_y, size_z)

    i_mid_z = int(size_z/2)
    
    data_2D_xy = my_reader.get_data_plane("Density",plane_type.xy, i_mid_z)
    x_grid_cen = my_reader.get_x_grid_cen()
    y_grid_cen = my_reader.get_y_grid_cen()
    
    
    
    plotter = data_plotter
    #plotter.plot_3D_slices(data_2D_xy, data_2D_xz, data_2D_yz,
    #                       x_grid, y_grid, z_grid, "plot_3D")
    plotter.plot_3D(data_2D_xy, x_grid_cen, y_grid_cen, "plot_3D")
    
    
def analysis_3D_projected() :
    # Here we do a joint plot of all 3 midplanes
    file_name = "output_step140.h5"
    full_name = path_data + "/" + file_name;
    my_reader = data_reader(full_name)    
    
    # Now, get the data in the 3 mid planes
    size_x, size_y, size_z = my_reader.get_grid_extent()
    print("Size of grid",size_x, size_y, size_z)

    i_mid_x = int(size_x/2)
    i_mid_y = int(size_y/2)
    i_mid_z = int(size_z/2)
    
    data_2D_yz = my_reader.get_data_plane("Density",plane_type.yz, i_mid_x)
    y_grid = my_reader.get_x_grid_left()
    z_grid = my_reader.get_y_grid_left()
    y_grid_cen = my_reader.get_x_grid_cen()
    z_grid_cen = my_reader.get_y_grid_cen()
    data_2D_xz = my_reader.get_data_plane("Density",plane_type.xz, i_mid_y)
    x_grid = my_reader.get_x_grid_left()
    x_grid_cen = my_reader.get_x_grid_cen()
    data_2D_xy = my_reader.get_data_plane("Density",plane_type.xy, i_mid_z)
    
    
    
    plotter = data_plotter
    #plotter.plot_3D_slices(data_2D_xy, data_2D_xz, data_2D_yz,
    #                       x_grid, y_grid, z_grid, "plot_3D")
    plotter.plot_3D_projected(data_2D_xy, data_2D_xz, data_2D_yz,
                    x_grid_cen, y_grid_cen, z_grid_cen, 
                    x_grid, y_grid, z_grid, "plot_3D_projeced.pdf")    
            
if __name__ == "__main__" :
    analysis_single_plane()
    #analysis_3D()
    #analysis_3D_projected()
