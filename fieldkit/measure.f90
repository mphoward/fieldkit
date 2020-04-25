!  Efficient Fortran methods for computing measures.
!
!! @author Michael P. Howard

!> @brief Calculates the Minkowski functionals for a 3D array.
!!
!! The Minkowski functionals (volume, surface area, integral mean
!! curvature, Euler characteristic) are computed for a domain
!! represented by a 3D *lattice*. The lattice is digitized into
!! 1s and 0s, and the active domain is all the 1s. The lattice
!! is 0-indexed, and the voxels are assumed to be cubes of size *a*.
!! The units of the returned functionals are implicitly in *a*.
!!
!! Additional details can be found in the reference from which
!! this routine was taken::
!!   K. Michielsen and H. De Raedt, Integral-geometry morphological
!!   analysis, Physics Report 347, 461-538 (2001).
!! @see minkowski_add_voxel
!!
!! @param[in]   lattice     3D array of integers.
!! @param[in]   Lx          Size of lattice in x.
!! @param[in]   Ly          Size of lattice in y.
!! @param[in]   Lz          Size of lattice in z.
!! @param[out]  volume      Volume of active lattice.
!! @param[out]  surface     Surface area of active lattice.
!! @param[out]  curvature   Mean curvature of active lattice.
!! @param[out]  euler       Euler characteristic of active lattice.
!!
subroutine minkowski(lattice,Lx,Ly,Lz,volume,surface,curvature,euler)
implicit none
integer, intent(in) :: Lx, Ly, Lz
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: lattice
integer, intent(out) :: volume, surface, curvature, euler

integer, dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: tmp
integer v,s,c,e
integer jx,jy,jz

! fill all tmp pixels with 0
tmp = 0

! loop over pixels and accumulate functionals
volume = 0
surface = 0
curvature = 0
euler = 0
do jz = 0,Lz-1
do jy = 0,Ly-1
do jx = 0,Lx-1
    if (lattice(jx,jy,jz) > 0) then ! active pixel
        call minkowski_add_voxel(tmp,Lx,Ly,Lz,jx,jy,jz,v,s,c,e)
        tmp(jx,jy,jz) = 1 ! add pixel to image
        volume = volume + v
        surface = surface + s
        curvature = curvature + c
        euler = euler + e
    end if
end do ! jx
end do ! jy
end do ! jz
end subroutine

!> @brief Calculates the change in Minkowski functionals on addition of a pixel.
!!
!! The changes in the Minkowski functionals (volume, surface area,
!! integral mean curvature, Euler characteristic) are computed from
!! adding a voxel at (*jx*,*jy*,*jz*) to the 3D *lattice*. The lattice
!! is 0-indexed, and the voxels are assumed to be cubes of size *a*.
!! The units of the returned functionals are implicitly in *a*.
!!
!! Reference equations can be found in Appendix B of::
!!   K. Michielsen and H. De Raedt, Integral-geometry morphological
!!   analysis, Physics Report 347, 461-538 (2001).
!! @see minkowski
!!
!! @param[in]   lattice     3D array of integers.
!! @param[in]   Lx          Size of lattice in x.
!! @param[in]   Ly          Size of lattice in y.
!! @param[in]   Lz          Size of lattice in z.
!! @param[in]   jx          Added voxel x-coordinate.
!! @param[in]   jy          Added voxel y-coordinate.
!! @param[in]   jz          Added voxel z-coordinate.
!! @param[out]  volume      Change in volume.
!! @param[out]  surface     Change in surface area.
!! @param[out]  curvature   Change in mean curvature.
!! @param[out]  euler       Change in Euler characteristic.
!!
subroutine minkowski_add_voxel(lattice,Lx,Ly,Lz,jx,jy,jz,volume,surface,curvature,euler)
implicit none
integer, intent(in) :: Lx, Ly, Lz, jx, jy, jz
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: lattice
integer, intent(out) :: volume, surface, curvature, euler

integer, parameter :: volume_body = 1   ! (a*a*a, where a is lattice displacement)
integer, parameter :: surface_body = -6 ! (-6*a*a, open body)
integer, parameter :: surface_face = 2  ! (2*a*a, open face)
integer, parameter :: curv_body = 3     ! (3*a, open body)
integer, parameter :: curv_face = -2    ! (-2*a, open face)
integer, parameter :: curv_edge = 1     ! (a, open line)
integer, parameter :: euler_body = -1   ! (open body)
integer, parameter :: euler_face = 1    ! (open face)
integer, parameter :: euler_edge = -1   ! (open line)
integer, parameter :: euler_vertex = 1  ! (vertices)

integer nfaces, nedges, nvert
integer i0,j0,k0,jxi,jyi,jzi,jyj,jzj,jzk
integer kc1,kc2,kc3,kc7,kc1kc4kc5

! accumulate faces, edges, vertices from adding pixel
nfaces = 0
nedges = 0
nvert = 0
do i0 = -1,1,2
    ! wrap offsets to account for periodic boundaries
    jxi = modulo(jx+i0,Lx)
    jyi = modulo(jy+i0,Ly)
    jzi = modulo(jz+i0,Lz)
    kc1 = 1-lattice(jxi,jy,jz)
    kc2 = 1-lattice(jx,jyi,jz)
    kc3 = 1-lattice(jx,jy,jzi)
    nfaces = nfaces + kc1 + kc2 + kc3 ! (B.5)
    do j0 = -1,1,2
        jyj = modulo(jy+j0,Ly)
        jzj = modulo(jz+j0,Lz)
        kc7 = 1-lattice(jx,jy,jzj)
        kc1kc4kc5 = kc1*(1-lattice(jxi,jyj,jz))*(1-lattice(jx,jyj,jz))
        nedges = nedges + kc1kc4kc5 + kc2*(1-lattice(jx,jyi,jzj))*kc7 + kc1*(1-lattice(jxi,jy,jzj))*kc7 ! (B.6)
        if (kc1kc4kc5.ne.0) then ! whole term (B.7) evaluates to 0 otherwise
            do k0 = -1,1,2
                jzk = modulo(jz+k0,Lz)
                nvert = nvert + (1-lattice(jxi,jy,jzk))*(1-lattice(jxi,jyj,jzk))*(1-lattice(jx,jyj,jzk))*(1-lattice(jx,jy,jzk)) ! (B.7, without kc1kc4kc5)
            end do !k0
        end if
    end do ! j0
end do ! i0

volume = volume_body
surface = surface_body + surface_face*nfaces
curvature = curv_body + curv_face*nfaces + curv_edge*nedges
euler = euler_body + euler_face*nfaces + euler_edge*nedges + euler_vertex*nvert
end subroutine
