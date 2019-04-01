!  Efficient Fortran methods for performing simulations.
!
!! @author Michael P. Howard

!> @brief Performs a simple lattice random walk.
!!
!! Every step, a walker chooses a random direction from the 6-connected
!! lattice. It steps in that direction if the node it is going to is also
!! in the domain. It is assumed that the walker starts from a node in the
!! domain.
!!
!! This method makes use of Fortran's random_number subroutine.
!! It needs to be appropriately seeded first using init_random_seed.
!!
!! @param[in]       domain      Integer (0 or 1) representation of the domain.
!! @param[in]       Lx          Size of domain in x.
!! @param[in]       Ly          Size of domain in y.
!! @param[in]       Lz          Size of domain in z.
!! @param[inout]    coord       Coordinates of the walkers in the domain.
!! @param[inout]    image       Image flags for the walkers.
!! @param[in]       N           Number of walkers.
!! @param[in]       steps       Number of steps to make.
!!
subroutine random_walk(domain,Lx,Ly,Lz,coord,image,N,steps)
implicit none
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: domain
integer, intent(in) :: Lx,Ly,Lz
integer, intent(inout), dimension(0:2,0:N-1) :: coord,image
integer, intent(in) :: N,steps

! random walk for 6-connected lattice
integer step,i,choice
integer, dimension(0:2) :: new_coord, new_image
do step = 0,steps-1
    do i = 0,N-1
        ! choose a random step and wrap
        new_coord = coord(:,i)
        new_image = image(:,i)
        call random_int(6,choice)
        select case (choice)
            case (0)
                new_coord(0) = new_coord(0)+1
                if (new_coord(0) >= Lx) then
                    new_coord(0) = new_coord(0) - Lx
                    new_image(0) = new_image(0) + 1
                end if
            case (1)
                new_coord(0) = new_coord(0)-1
                if (new_coord(0) < 0) then
                    new_coord(0) = new_coord(0) + Lx
                    new_image(0) = new_image(0) - 1
                end if
            case (2)
                new_coord(1) = new_coord(1)+1
                if (new_coord(1) >= Ly) then
                    new_coord(1) = new_coord(1) - Ly
                    new_image(1) = new_image(1) + 1
                end if
            case (3)
                new_coord(1) = new_coord(1)-1
                if (new_coord(1) < 0) then
                    new_coord(1) = new_coord(1) + Ly
                    new_image(1) = new_image(1) - 1
                end if
            case (4)
                new_coord(2) = new_coord(2)+1
                if (new_coord(2) >= Lz) then
                    new_coord(2) = new_coord(2) - Lz
                    new_image(2) = new_image(2) + 1
                end if
            case (5)
                new_coord(2) = new_coord(2)-1
                if (new_coord(2) < 0) then
                    new_coord(2) = new_coord(2) + Lz
                    new_image(2) = new_image(2) - 1
                end if
        end select

        ! accept move if it is in the domain, otherwise stay put
        if (domain(new_coord(0),new_coord(1),new_coord(2)) == 1) then
            coord(:,i) = new_coord
            image(:,i) = new_image
        end if
    enddo ! i
enddo ! step
end subroutine

!> @brief Initialize the random number generator.
subroutine init_random_seed()
implicit none
call random_seed()
end subroutine

!> @brief Draws a random integer from [0,n).
!!
!! This method uses random_number, so ensure that it has
!! been seeded properly.
!!
!! @param[in]   n       Maximum integer.
!! @param[out]  choice  Random integer lying in [0,n).
!!
subroutine random_int(n,choice)
implicit none
integer, intent(in) :: n
integer, intent(out) :: choice

real*8 r
call random_number(r)
choice = floor(n*r)
end subroutine

!> @brief Computes the mean-square displacement of a trajectory.
!!
!! @param[in]   traj    Trajectory to analyze.
!! @param[in]   runs    Number of runs in the trajectory.
!! @param[in]   N       Number of particles in the trajectory.
!! @param[out]  rsq     Component-wise mean-square displacement.
!! @param[in]   window  Time window for computing the MSD.
!!
!! The traj should be a 3xrunsxN multidimensional array. The
!! msd is evaluated over the window (inclusive), so the shape of
!! rsq is 3x(window+1). (The first entry are the trivial zeros.)
!!
subroutine msd(traj,runs,N,rsq,window)
implicit none
real*8, intent(in), dimension(0:2,0:runs-1,0:N-1) :: traj
integer, intent(in) :: runs,N
real*8, intent(out), dimension(0:2, 0:window) :: rsq
integer, intent(in) :: window

! internal variables for computing and accumulating the msd
real*8, dimension(0:2) :: r0,dr
integer i,t0,dt,ax
integer, dimension(0:window) :: counts

! compute msd by iterating through trajectory
rsq = 0.
counts = 0
do i = 0,N-1
    do t0 = 0,runs-2
        r0 = traj(:,t0,i)
        do dt = 1,min(window,runs-1-t0)
            dr = traj(:,t0+dt,i) - r0
            do ax = 0,2
                rsq(ax,dt) = rsq(ax,dt) + dr(ax)*dr(ax)
            enddo
            counts(dt) = counts(dt) + 1
        enddo
    enddo
enddo

! normalize by number of counts
do t0 = 0,window
    if (counts(t0) > 0) then
        rsq(:,t0) = rsq(:,t0) / counts(t0)
    endif
enddo
end subroutine
