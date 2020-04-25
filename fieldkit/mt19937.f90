! Fortran translation of Mersenne Twister pseudo-random number generator.
!
!! @author Michael P. Howard
!
! The original code is modified and used under the following license:
!
!   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
!   All rights reserved.
!
!   Redistribution and use in source and binary forms, with or without
!   modification, are permitted provided that the following conditions
!   are met:
!
!     1. Redistributions of source code must retain the above copyright
!        notice, this list of conditions and the following disclaimer.
!
!     2. Redistributions in binary form must reproduce the above copyright
!        notice, this list of conditions and the following disclaimer in the
!        documentation and/or other materials provided with the distribution.
!
!     3. The names of its contributors may not be used to endorse or promote
!        products derived from this software without specific prior written
!        permission.
!
!   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
!   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
!   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
!   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
!   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
!   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
!   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
!   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
!   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
!   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! The original paper can be found here:
!
!   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/values/ARTICLES/mt.pdf
!
! Note this paper uses a different initialization procedure, which was later
! shown to have some issues.

module mt19937
    implicit none

    private
    public :: mt19937_t, mt19937_seed, mt19937_int32, mt19937_uniform, &
              mt19937_uniform_int, mt19937_uniform_real

    ! values required for manipulating the MT state
    integer(kind=4),parameter :: N = 624
    integer(kind=4),parameter :: M = 397
    integer(kind=4),parameter :: LOWER_MASK = 2147483647
    integer(kind=4),parameter :: UPPER_MASK = -LOWER_MASK-1
    integer(kind=4),parameter,dimension(0:1) :: MAG01 = (/0,-1727483681/)

    !> @brief State of the random number generator
    !!
    !! This holds a vector of values created when the generator is advanced.
    !! The index points into this list of values (0-indexed). Restoring this
    !! state would be sufficient to restart the generator exactly.
    type :: mt19937_t
        integer(kind=4),dimension(0:N-1) :: values
        integer(kind=4) :: idx = N
    end type mt19937_t

contains

    !> @ brief Set seed for the random number generator.
    !!
    !! The initial state of the generator is set using a linear
    !! congruential generator. The index of the state is set to its end
    !! so that the next call will update the state. This method **must**
    !! be called before using the state of the generator.
    !!
    !! @param[inout]    state   State of the random number generator.
    !! @param[in]       seed    Seed for the generator.
    !!
    subroutine mt19937_seed(state, seed)
        type(mt19937_t), intent(inout) :: state
        integer(kind=4), intent(in) :: seed
        integer(kind=4) i

        state%idx = N
        state%values(0) = iand(seed,-1)
        do i=1,N-1
            state%values(i) = iand(1812433253*ieor(state%values(i-1),ishft(state%values(i-1),-30))+i,-1)
        end do
    end subroutine mt19937_seed

    !!> @brief Generate a random 32-bit signed integer.
    !!
    !! A random integer is drawn using the MT19937 algorithm. The state of
    !! the generator is advanced if new values are required; otherwise, values
    !! are pulled from the cached state. The returned value is tempered for use.
    !!
    !! @param[inout]    state   State of the random number generator.
    !! @param[out]      value   Random 32-bit signed integer.
    !!
    subroutine mt19937_int32(state, value)
        type(mt19937_t), intent(inout) :: state
        integer(kind=4), intent(out) :: value
        integer(kind=4) i,y

        ! advance state if we have run out of values
        if (state%idx >= N) then
            ! section 1
            do i=0,N-M-1
                y = ior(iand(state%values(i),UPPER_MASK),iand(state%values(i+1),LOWER_MASK))
                state%values(i) = ieor(ieor(state%values(i+M),ishft(y,-1)),MAG01(iand(y,1)))
            end do

            ! section 2
            do i=N-M,N-2
                y = ior(iand(state%values(i),UPPER_MASK),iand(state%values(i+1),LOWER_MASK))
                state%values(i) = ieor(ieor(state%values(i+(M-N)),ishft(y,-1)),MAG01(iand(y,1)))
            end do

            ! final section
            y = ior(iand(state%values(N-1),UPPER_MASK),iand(state%values(0),LOWER_MASK))
            state%values(i) = ieor(ieor(state%values(M-1),ishft(y,-1)),MAG01(iand(y,1)))

            ! reset counter
            state%idx = 0
        endif

        ! temper value
        y = state%values(state%idx)
        state%idx = state%idx + 1
        y = ieor(y,ishft(y,-11))
        y = ieor(y,iand(ishft(y,7),-1658038656))
        y = ieor(y,iand(ishft(y,15),-272236544))
        y = ieor(y,ishft(y,-18))

        ! return
        value = y
    end subroutine mt19937_int32

    !!> @brief Generate a uniform random real in [0,1) with 32-bits of randomness.
    !!
    !! The random real is generated by drawing an integer and converting it to the
    !! [0,1) range. Note that this generates only 32-bits of randomness. If additional
    !! bits are required, a method drawing 53-bits from two 32-bit integers can be
    !! implemented.
    !!
    !! @param[inout]    state   State of the random number generator.
    !! @param[out]      value   Random real(8) in [0,1).
    !!
    subroutine mt19937_uniform(state, value)
        type(mt19937_t), intent(inout) :: state
        real(kind=8), intent(out) :: value
        integer(kind=4) y

        call mt19937_int32(state,y)

        ! convert int to double
        value = 0.5d0+dble(y)/4294967296.0d0
    end subroutine mt19937_uniform

    !!> @brief Generate a uniform random int in [0,n).
    !!
    !! A random integer in [0,n) is generated by drawing a uniform real, then
    !! scaling and flooring it into this range.
    !!
    !! @param[inout]    state   State of the random number generator.
    !! @param[out]      value   Random integer in [0,n).
    !! @param[n]        n       Maximum value (exclusive).
    !!
    subroutine mt19937_uniform_int(state, value, n)
        type(mt19937_t), intent(inout) :: state
        integer(kind=4), intent(out) :: value
        integer(kind=4), intent(in) :: n
        real(kind=8) :: r

        call mt19937_uniform(state,r)
        value = floor(n*r)
    end subroutine mt19937_uniform_int

    !!> @brief Generate a uniform random real in [0,n) with 32-bits of randomness.
    !!
    !! A random real in [0,n) is generated by scaling a uniform real. Note that this
    !! real still has 32-bits of randomness.
    !!
    !! @param[inout]    state   State of the random number generator.
    !! @param[out]      value   Random real(8) in [0,n).
    !! @param[n]        n       Maximum value (exclusive).
    !!
    subroutine mt19937_uniform_real(state, value, n)
        type(mt19937_t), intent(inout) :: state
        real(kind=8), intent(out) :: value
        real(kind=8), intent(in) :: n

        call mt19937_uniform(state, value)
        value = n*value
    end subroutine mt19937_uniform_real

end module mt19937
