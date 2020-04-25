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
!! @param[in]       domain      Integer (0 or 1) representation of the domain.
!! @param[in]       Lx          Size of domain in x.
!! @param[in]       Ly          Size of domain in y.
!! @param[in]       Lz          Size of domain in z.
!! @param[inout]    coord       Coordinates of the walkers in the domain.
!! @param[inout]    image       Image flags for the walkers.
!! @param[out]      traj        Unwrapped trajectory.
!! @param[in]       N           Number of walkers.
!! @param[in]       steps       Number of steps to make per run.
!! @param[in]       runs        Numbers of runs to make.
!! @param[in]       seed        Seed to mt19937 PRNG.
!! @param[in]       Nt          Number of parallel threads.
!!
subroutine random_walk(domain,Lx,Ly,Lz,coord,image,traj,N,steps,runs,seed,Nt)
!$f2py threadsafe
use omp_lib
use mt19937
implicit none
integer, intent(in) :: Lx,Ly,Lz,N,steps,runs,seed,Nt
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: domain
integer, intent(inout), dimension(0:2,0:N-1) :: coord,image
integer, intent(out), dimension(0:2,0:N-1,0:runs-1) :: traj

! random walk for 6-connected lattice
integer i,run,step,choice
integer, dimension(0:2) :: cur_coord, cur_image, new_coord, new_image, box
type(mt19937_t) :: rng

box = (/Lx,Ly,Lz/)

!$omp parallel default(none), private(i,run,step,choice,cur_coord,cur_image,new_coord,new_image,rng), &
!$omp shared(domain,coord,image,traj,N,steps,runs,seed,box), num_threads(Nt)

! seed rng for each thread using different value
call mt19937_seed(rng, seed+omp_get_thread_num())

!$omp do
do i = 0,N-1
    cur_coord = coord(:,i)
    cur_image = image(:,i)
    do run = 0,runs-1
        ! save position of walker at beginning of run
        traj(:,i,run) = cur_coord + cur_image * box
        do step = 0,steps-1
            ! choose a random step and wrap
            new_coord = cur_coord
            new_image = cur_image
            call mt19937_uniform_int(rng,choice,6)
            call take_step(choice,box,new_coord,new_image)
            ! accept move if it is in the domain, otherwise stay put
            if (domain(new_coord(0),new_coord(1),new_coord(2)) == 1) then
                cur_coord = new_coord
                cur_image = new_image
            end if
        enddo ! step
    enddo ! run
    ! save walker state at end of trajectory
    coord(:,i) = cur_coord
    image(:,i) = cur_image
enddo ! i
!$omp end do
!$omp end parallel
end subroutine

!> @brief Performs a biased random walk using rejection.
!!
!! Every step, a walker chooses a direction from the 6-connected lattice
!! with weight given by the transition probabilities. As a sanity check,
!! it is assumed that the walker starts from a node in the domain, and
!! that only steps onto nodes that are also in the domain are accepted.
!! This serves as a guard against potential round-off error in the cumulative
!! sum calculation.
!!
!! @param[in]       domain      Integer (0 or 1) representation of the domain.
!! @param[in]       Lx          Size of domain in x.
!! @param[in]       Ly          Size of domain in y.
!! @param[in]       Lz          Size of domain in z.
!! @param[in]       cumprob     Cumulative probabilities of transition to adjacent sites.
!! @param[inout]    coord       Coordinates of the walkers in the domain.
!! @param[inout]    image       Image flags for the walkers.
!! @param[out]      traj        Unwrapped trajectory.
!! @param[in]       N           Number of walkers.
!! @param[in]       steps       Number of steps to make per run.
!! @param[in]       runs        Numbers of runs to make.
!! @param[in]       seed        Seed to mt19937 PRNG.
!! @param[in]       Nt          Number of parallel threads.
!!
subroutine biased_walk(domain,Lx,Ly,Lz,cumprob,coord,image,traj,N,steps,runs,seed,Nt)
!$f2py threadsafe
use omp_lib
use mt19937
implicit none
integer, intent(in) :: Lx,Ly,Lz,N,steps,runs,seed,Nt
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: domain
real(kind=8), intent(in), dimension(0:5,0:Lx-1,0:Ly-1,0:Lz-1) :: cumprob
integer, intent(inout), dimension(0:2,0:N-1) :: coord,image
integer, intent(out), dimension(0:2,0:N-1,0:runs-1) :: traj

! random walk for 6-connected lattice
integer i,run,step,choice
integer, dimension(0:2) :: cur_coord, cur_image, new_coord, new_image, box
type(mt19937_t) :: rng
real(kind=8) :: rval

box = (/Lx,Ly,Lz/)

!$omp parallel default(none), private(i,run,step,choice,cur_coord,cur_image,new_coord,new_image,rng,rval), &
!$omp shared(domain,cumprob,coord,image,traj,N,steps,runs,seed,box), num_threads(Nt)

! seed rng for each thread using different value
call mt19937_seed(rng, seed+omp_get_thread_num())

!$omp do
do i = 0,N-1
    cur_coord = coord(:,i)
    cur_image = image(:,i)
    do run = 0,runs-1
        ! save position of walker at beginning of run
        traj(:,i,run) = cur_coord + cur_image * box
        do step = 0,steps-1
            ! choose a random step and wrap
            new_coord = cur_coord
            new_image = cur_image
            call mt19937_uniform(rng,rval)
            do choice = 0,5
                if (rval <= cumprob(choice,cur_coord(0),cur_coord(1),cur_coord(2))) then
                    call take_step(choice,box,new_coord,new_image)
                    ! sanity check: only accept move if it is in the domain
                    if (domain(new_coord(0),new_coord(1),new_coord(2)) == 1) then
                        cur_coord = new_coord
                        cur_image = new_image
                    end if
                    ! move made, break choice loop
                    exit
                end if
            end do ! choice
        enddo ! step
    enddo ! run
    ! save walker state at end of trajectory
    coord(:,i) = cur_coord
    image(:,i) = cur_image
enddo ! i
!$omp end do
!$omp end parallel
end subroutine

!> @brief Performs a biased random walk using rejection-free kMC.
!!
!! Every step, a walker chooses a direction from the 6-connected lattice.
!! The move is unconditionally accepted if it remains in the domain, and
!! time is advanced by a random value chosen from an exponential distribution.
!! If for some reason a move would advance the walker outside the domain, the
!! move is rejected but the time is still advanced. Such moves are effectively
!! transitions to the current position, and so are included in the time counter.
!!
!! @param[in]       domain      Integer (0 or 1) representation of the domain.
!! @param[in]       Lx          Size of domain in x.
!! @param[in]       Ly          Size of domain in y.
!! @param[in]       Lz          Size of domain in z.
!! @param[in]       cumrate     Cumulative transition rates to adjacent sites.
!! @param[inout]    coord       Coordinates of the walkers in the domain.
!! @param[inout]    image       Image flags for the walkers.
!! @param[inout]    time        Current time for the walkers.
!! @param[in]       sampleat    Time points to sample the trajectory.
!! @param[out]      traj        Unwrapped trajectory.
!! @param[in]       N           Number of walkers.
!! @param[in]       runs        Numbers of runs to make.
!! @param[in]       steps       Maximum number of steps to attempt.
!! @param[in]       seed        Seed to mt19937 PRNG.
!! @param[in]       Nt          Number of parallel threads.
!!
subroutine kmc(domain,Lx,Ly,Lz,cumrate,coord,image,time,sampleat,traj,N,runs,steps,seed,Nt)
!$f2py threadsafe
use omp_lib
use mt19937
implicit none
integer, intent(in) :: Lx,Ly,Lz,N,runs,steps,seed,Nt
integer, intent(in), dimension(0:Lx-1,0:Ly-1,0:Lz-1) :: domain
real(kind=8), intent(in), dimension(0:5,0:Lx-1,0:Ly-1,0:Lz-1) :: cumrate
integer, intent(inout), dimension(0:2,0:N-1) :: coord,image
real(kind=8), intent(inout), dimension(0:N-1) :: time
real(kind=8), intent(in), dimension(0:runs-1) :: sampleat
integer, intent(out), dimension(0:2,0:N-1,0:runs-1) :: traj

! random walk for 6-connected lattice
integer i,step,run,choice
integer, dimension(0:2) :: cur_coord, cur_image, new_coord, new_image, box
type(mt19937_t) :: rng
real(kind=8) :: rval,cur_time,new_time,max_rate

box = (/Lx,Ly,Lz/)

!$omp parallel default(none), &
!$omp private(i,step,run,choice,cur_coord,cur_image,new_coord,new_image,rng,rval,cur_time,new_time,max_rate), &
!$omp shared(domain,cumrate,coord,image,time,sampleat,traj,N,runs,steps,seed,box), num_threads(Nt)

! seed rng for each thread using different value
call mt19937_seed(rng, seed+omp_get_thread_num())

!$omp do
do i = 0,N-1
    cur_coord = coord(:,i)
    cur_image = image(:,i)
    cur_time = time(i)

    ! advance the sample counter until we find one we haven't already covered
    run = 0
    do while (run < runs .AND. cur_time > sampleat(run))
        traj(:,i,run) = 2147483647
        run = run + 1
    end do

    ! attempt to keep stepping while we haven't finished sampling
    step = 0
    do while (step < steps .AND. run < runs)
        ! choose a move based on the rates
        max_rate = cumrate(5,cur_coord(0),cur_coord(1),cur_coord(2))
        call mt19937_uniform_real(rng,rval,max_rate)
        do choice = 0,5
            if (rval <= cumrate(choice,cur_coord(0),cur_coord(1),cur_coord(2))) then
                exit
            end if
        end do ! choice

        ! take step
        new_coord = cur_coord
        new_image = cur_image
        call take_step(choice,box,new_coord,new_image)
        ! sanity check: only accept move if it is in the domain, otherwise revert.
        ! this implies a move to stay on the lattice site, so time counter should advance.
        if (domain(new_coord(0),new_coord(1),new_coord(2)) == 0) then
            new_coord = cur_coord
            new_image = cur_image
        end if

        ! advance time according to waiting time distribution
        call mt19937_uniform(rng,rval)
        new_time = cur_time - log(1.-rval)/max_rate

        ! record the samples that were covered during this waiting time
        do while (run < runs .AND. new_time > sampleat(run))
            traj(:,i,run) = cur_coord + cur_image * box
            run = run + 1
        end do

        ! finalize the move
        cur_coord = new_coord
        cur_image = new_image
        cur_time = new_time
        step = step + 1
    end do ! steps & runs

    ! save walker state at end of trajectory
    coord(:,i) = cur_coord
    image(:,i) = cur_image
    time(i) = cur_time
enddo ! i
!$omp end do
!$omp end parallel
end subroutine

!> @brief Take a step along a direction with periodic boundary conditions.
!!
!! The coordinates and images are wrapped using the box, which runs from [0,L).
!!
!! @param[in]       dir         The direction to step in.
!! @param[in]       box         The box shape.
!! @param[inout]    coord       Coordinates of the walker.
!! @param[inout]    image       Image flags for the walker.
!!
subroutine take_step(dir,box,coord,image)
implicit none
integer, intent(in) :: dir
integer, dimension(0:2), intent(in) :: box
integer, dimension(0:2), intent(inout) :: coord,image

select case (dir)
    case (0)
        coord(0) = coord(0)+1
        if (coord(0) >= box(0)) then
            coord(0) = coord(0) - box(0)
            image(0) = image(0) + 1
        end if
    case (1)
        coord(0) = coord(0)-1
        if (coord(0) < 0) then
            coord(0) = coord(0) + box(0)
            image(0) = image(0) - 1
        end if
    case (2)
        coord(1) = coord(1)+1
        if (coord(1) >= box(1)) then
            coord(1) = coord(1) - box(1)
            image(1) = image(1) + 1
        end if
    case (3)
        coord(1) = coord(1)-1
        if (coord(1) < 0) then
            coord(1) = coord(1) + box(1)
            image(1) = image(1) - 1
        end if
    case (4)
        coord(2) = coord(2)+1
        if (coord(2) >= box(2)) then
            coord(2) = coord(2) - box(2)
            image(2) = image(2) + 1
        end if
    case (5)
        coord(2) = coord(2)-1
        if (coord(2) < 0) then
            coord(2) = coord(2) + box(2)
            image(2) = image(2) - 1
        end if
end select
end subroutine

!> @brief Computes the mean-square displacement of a trajectory.
!!
!! @param[in]   traj    Trajectory to analyze.
!! @param[in]   runs    Number of runs in the trajectory.
!! @param[in]   N       Number of particles in the trajectory.
!! @param[out]  rsq     Component-wise mean-square displacement.
!! @param[in]   window  Time window for computing the MSD.
!! @param[in]   every   Number of runs between time origins.
!! @param[in]   Nt      Number of parallel threads.
!!
!! The traj should be a 3xrunsxN multidimensional array. The
!! msd is evaluated over the window (inclusive), so the shape of
!! rsq is 3x(window+1). (The first entry are the trivial zeros.)
!!
subroutine msd(traj,runs,N,rsq,window,every,Nt)
!$f2py threadsafe
use omp_lib
implicit none
integer, intent(in) :: runs,N,window,every,Nt
real(kind=8), intent(in), dimension(0:2,0:runs-1,0:N-1) :: traj
real(kind=8), intent(out), dimension(0:2, 0:window) :: rsq

! internal variables for computing and accumulating the msd
real(kind=8), dimension(0:2) :: r0,dr
integer i,t0,dt,ax,tid
integer, dimension(0:window,0:Nt-1) :: counts_
real(kind=8), dimension(0:2,0:window,0:Nt-1) :: rsq_

! compute msd by iterating through trajectory
rsq_ = 0.
counts_ = 0
!$omp parallel default(none), private(r0,dr,i,t0,dt,ax,tid), &
!$omp shared(counts_,rsq_,traj,runs,N,window,every), num_threads(Nt)
tid = omp_get_thread_num()
!$omp do
do i = 0,N-1
    do t0 = 0,runs-2,every
        r0 = traj(:,t0,i)
        do dt = 1,min(window,runs-1-t0)
            dr = traj(:,t0+dt,i) - r0
            do ax = 0,2
                rsq_(ax,dt,tid) = rsq_(ax,dt,tid) + dr(ax)*dr(ax)
            enddo
            counts_(dt,tid) = counts_(dt,tid) + 1
        enddo
    enddo
enddo
!$omp end do
!$omp end parallel

! reduce over threads
do i = 1,Nt-1
    counts_(:,0) = counts_(:,0) + counts_(:,i)
    rsq_(:,:,0) = rsq_(:,:,0) + rsq_(:,:,i)
end do

! output using normalization
do t0 = 0,window
    if (counts_(t0,0) > 0) then
        rsq(:,t0) = rsq_(:,t0,0) / counts_(t0,0)
    else
        rsq(:,t0) = 0.
    endif
enddo
end subroutine

!> @brief Computes the mean-square displacement of a trajectory with origin binning.
!!
!! @param[in]   traj    Trajectory to analyze.
!! @param[in]   runs    Number of runs in the trajectory.
!! @param[in]   N       Number of particles in the trajectory.
!! @param[in]   axis    Coordinate to use for binning.
!! @param[in]   bins    Number of bins along axis.
!! @param[in]   lo      Lower bound for binning.
!! @param[in]   hi      Upper bound for binning.
!! @param[out]  rsq     Component-wise mean-square displacement for each origin.
!! @param[in]   window  Time window for computing the MSD.
!! @param[in]   every   Number of runs between time origins.
!!
!! The traj should be a 3xrunsxN multidimensional array. The
!! msd is evaluated over the window (inclusive), so the shape of
!! rsq is 3x(window+1)xbins. (The first entry are the trivial zeros.)
!!
subroutine msd_binned(traj,runs,N,axis,bins,lo,hi,rsq,window,every)
implicit none
integer, intent(in) :: runs,N,axis,bins,window,every
real(kind=8), intent(in), dimension(0:2,0:runs-1,0:N-1) :: traj
real(kind=8), intent(in) :: lo,hi
real(kind=8), intent(out), dimension(0:2,0:window,0:bins-1) :: rsq

! internal variables for computing and accumulating the msd
real(kind=8) inv_bin_width
integer bin0
real(kind=8), dimension(0:2) :: r0,dr
integer i,t0,dt,ax
integer, dimension(0:window,0:bins-1) :: counts

! compute msd by iterating through trajectory
inv_bin_width = bins/(hi-lo)
rsq = 0.
counts = 0
do i = 0,N-1
    do t0 = 0,runs-2,every
        r0 = traj(:,t0,i)
        bin0 = floor((r0(axis)-lo)*inv_bin_width)

        ! silently ignore particles starting out of bin range
        if (bin0 >= 0 .AND. bin0 < bins) then
            do dt = 1,min(window,runs-1-t0)
                dr = traj(:,t0+dt,i) - r0
                do ax = 0,2
                    rsq(ax,dt,bin0) = rsq(ax,dt,bin0) + dr(ax)*dr(ax)
                enddo
                counts(dt,bin0) = counts(dt,bin0) + 1
            enddo
        endif
    enddo
enddo

! normalize by number of counts
do i = 0,bins-1
    do t0 = 0,window
        if (counts(t0,i) > 0) then
            rsq(:,t0,i) = rsq(:,t0,i) / counts(t0,i)
        endif
    enddo
enddo
end subroutine

!> @brief Computes the mean-square displacement of a trajectory with origin binning.
!!
!! @param[in]   traj    Trajectory to analyze.
!! @param[in]   runs    Number of runs in the trajectory.
!! @param[in]   N       Number of particles in the trajectory.
!! @param[in]   axis    Coordinate to use for binning.
!! @param[in]   bins    Number of bins along axis.
!! @param[in]   lo      Lower bound for binning.
!! @param[in]   hi      Upper bound for binning.
!! @param[out]  rsq     Component-wise mean-square displacement for each bin.
!! @param[out]  counts  Number of particles that survive in each bin.
!! @param[in]   window  Time window for computing the MSD.
!! @param[in]   every   Number of runs between time origins.
!!
!! The traj should be a 3xrunsxN multidimensional array. The
!! msd is evaluated over the window (inclusive), so the shape of
!! rsq is 3x(window+1)xbins. (The first entry are the trivial zeros.)
!!
subroutine msd_survival(traj,runs,N,axis,bins,lo,hi,rsq,counts,window,every)
implicit none
integer, intent(in) :: runs,N,axis,bins,window,every
real(kind=8), intent(in), dimension(0:2,0:runs-1,0:N-1) :: traj
real(kind=8), intent(in) :: lo,hi
real(kind=8), intent(out), dimension(0:2,0:window,0:bins-1) :: rsq
integer, intent(out), dimension(0:window,0:bins-1) :: counts

! internal variables for computing and accumulating the msd
real(kind=8) inv_bin_width
integer bin0,bin1
real(kind=8), dimension(0:2) :: r0,r1,dr
integer i,t0,dt,ax

! compute msd by iterating through trajectory
inv_bin_width = bins/(hi-lo)
rsq = 0.
counts = 0
do i = 0,N-1
    do t0 = 0,runs-2,every
        ! break loop if window does not fit
        if (t0+window >= runs) then
            exit
        endif

        ! particle that starts in the bin
        r0 = traj(:,t0,i)
        bin0 = floor((r0(axis)-lo)*inv_bin_width)
        if (bin0 >= 0 .AND. bin0 < bins) then
            counts(0,bin0) = counts(0,bin0) + 1
        else
            ! ignore time origin if particle starts out of bin range
            cycle
        endif

        ! evaluate MSD (dt guaranteed to stay in valid range)
        do dt = 1,window
            ! particle must stay in bin to keep accumulating
            r1 = traj(:,t0+dt,i)
            bin1 = floor((r1(axis)-lo)*inv_bin_width)
            if (bin1 .ne. bin0) then
                exit
            endif

            dr = r1 - r0
            do ax = 0,2
                if (ax .ne. axis) then
                    rsq(ax,dt,bin0) = rsq(ax,dt,bin0) + dr(ax)*dr(ax)
                endif
            enddo
            counts(dt,bin0) = counts(dt,bin0) + 1
        enddo
    enddo
enddo

! normalize by number of counts **starting** in the bin
do i = 0,bins-1
    do t0 = 0,window
        if (counts(0,i) > 0) then
            rsq(:,t0,i) = rsq(:,t0,i) / counts(0,i)
        endif
    enddo
enddo
end subroutine

!> @brief Computes the axial mean-square displacement of a trajectory in a cylinder with radial binning.
!!
!! @param[in]   traj    Trajectory to analyze.
!! @param[in]   runs    Number of runs in the trajectory.
!! @param[in]   N       Number of particles in the trajectory.
!! @param[in]   bins    Number of bins along axis.
!! @param[in]   lo      Lower bound for binning.
!! @param[in]   hi      Upper bound for binning.
!! @param[out]  rsq     Axial mean-square displacement for each bin.
!! @param[out]  counts  Number of particles that survive in each bin.
!! @param[in]   window  Time window for computing the MSD.
!! @param[in]   every   Number of runs between time origins.
!!
!! radial and axial should be runsxN multidimensional arrays. The
!! msd is evaluated over the window (inclusive), so the shape of
!! rsq is (window+1)xbins. (The first entry are the trivial zeros.)
!!
subroutine msd_survival_cylinder(radial,axial,runs,N,bins,lo,hi,rsq,counts,window,every)
implicit none
integer, intent(in) :: runs,N,bins,window,every
real(kind=8), intent(in), dimension(0:runs-1,0:N-1) :: radial,axial
real(kind=8), intent(in) :: lo,hi
real(kind=8), intent(out), dimension(0:window,0:bins-1) :: rsq
integer, intent(out), dimension(0:window,0:bins-1) :: counts

! internal variables for computing and accumulating the msd
real(kind=8) inv_bin_width
integer bin0,bin1
real(kind=8) r0,r1,z0,z1,dz
integer i,t0,t1,dt

! compute msd by iterating through trajectory
inv_bin_width = bins/(hi-lo)
rsq = 0.
counts = 0
do i = 0,N-1
    do t0 = 0,runs-2,every
        ! break loop if window does not fit
        if (t0+window >= runs) then
            exit
        endif

        ! particle that starts in the bin
        r0 = radial(t0,i)
        z0 = axial(t0,i)
        bin0 = floor((r0-lo)*inv_bin_width)
        if (bin0 >= 0 .AND. bin0 < bins) then
            counts(0,bin0) = counts(0,bin0) + 1
        else
            ! ignore time origin if particle starts out of bin range
            cycle
        endif

        ! evaluate MSD (dt guaranteed to stay in valid range)
        do dt = 1,window
            t1 = t0 + dt
            ! particle must stay in bin to keep accumulating
            r1 = radial(t1,i)
            bin1 = floor((r1-lo)*inv_bin_width)
            if (bin1 .ne. bin0) then
                exit
            endif

            z1 = axial(t1,i)
            dz = z1-z0
            rsq(dt,bin0) = rsq(dt,bin0) + dz*dz
            counts(dt,bin0) = counts(dt,bin0) + 1
        enddo
    enddo
enddo

! normalize by number of counts **starting** in the bin
do i = 0,bins-1
    do t0 = 0,window
        if (counts(0,i) > 0) then
            rsq(t0,i) = rsq(t0,i) / counts(0,i)
        endif
    enddo
enddo
end subroutine
