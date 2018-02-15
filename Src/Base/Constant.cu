#include  <Base/Math.hpp>
#include <Base/Constant.hpp>
#include <algorithm>
#include <stdexcept>

namespace Impl {
    __constant__ unsigned char memory[blockNum*blockSize];

    class SegTree final {
    private:
        static constexpr auto treeSize = (blockNum + 1) << 2;
        uint mSiz[treeSize], mLsiz[treeSize], mRsiz[treeSize];
        u8 mFlag[treeSize];
#define LS l,m,id<<1
#define RS m+1,r,id<<1|1
        void build(const uint l, const uint r, const uint id) {
            mSiz[id] = mLsiz[id] = mRsiz[id] = r - l + 1;
            mFlag[id] = 2;
            if (l != r) {
                const auto m = (l + r) >> 1;
                build(LS);
                build(RS);
            }
        }
        void update(const uint l, const uint r, const uint id) {
            const auto m = (l + r) >> 1;
            mLsiz[id] = mLsiz[id << 1];
            if (mLsiz[id] == m - l + 1)mLsiz[id] += mLsiz[id << 1 | 1];
            mRsiz[id] = mRsiz[id << 1|1];
            if (mRsiz[id] == r - m)mRsiz[id] += mRsiz[id << 1];
            mSiz[id] = std::max(std::max(mSiz[id << 1], mSiz[id << 1 | 1]), mRsiz[id << 1] + mLsiz[id<<1|1]);
        }

        void color(const uint l, const uint r, const uint id, const u8 mark) {
            mFlag[id] = mark;
            mSiz[id] = mLsiz[id] = mRsiz[id] = (mark ? r - l + 1U : 0U);
        }

        void push(const uint l, const uint r, const uint id) {
            if (mFlag[id] != 2) {
                const auto m = (l + r) >> 1;
                color(LS, mFlag[id]);
                color(RS, mFlag[id]);
                mFlag[id] = 2;
            }
        }

        void modify(const uint l, const uint r, const uint id, const uint nl, const uint nr, const u8 mark) {
            if (nl <= l && r <= nr) color(l, r, id, mark);
            else {
                push(l, r, id);
                const auto m = (l + r) >> 1;
                if (nl <= m)modify(LS, nl, nr, mark);
                if (m < nr)modify(RS, nl, nr, mark);
                update(l, r, id);
            }
        }

        uint query(const uint l, const uint r, const uint id, const uint size) {
            if (l == r)return l;
            push(l, r, id);
            const auto m = (l + r) >> 1;
            if (mSiz[id << 1] >= size)return query(LS, size);
            if (mRsiz[id << 1] + mLsiz[id << 1 | 1] >= size)return m - mRsiz[id << 1] + 1;
            return query(RS, size);
        }
#undef LS
#undef RS
    public:
        SegTree(){
            build(1, blockNum, 1);
        }
        uint allocSpace(uint size) {
            if (mSiz[1] >= size) {
                auto l = query(1, blockNum, 1, size);
                modify(1,blockNum,1,l,l+size-1,0);
                return l;
            }
            throw std::runtime_error("Out of memory.");
        }
        void returnSpace(const uint off, const uint size) {
            modify(1, blockNum, 1, off, off + size - 1, 1);
        }
    };

    SegTree& getPool() {
        static thread_local SegTree pool;
        return pool;
    }

    auto getAddress() {
        static thread_local void* address=nullptr;
        if(address==nullptr)
            checkError(cudaGetSymbolAddress(&address, memory));
        return static_cast<unsigned char*>(address);
    }

    void* constantAlloc(const unsigned int size) {
        const auto req = calcSize(size,blockSize);
        const auto off = getPool().allocSpace(req);
        if (off == 0)return nullptr;
        return getAddress()+(off-1)*blockSize;
    }

    void constantFree(void* address, const unsigned int size) {
        const auto req = calcSize(size, blockSize);
        const auto off = (static_cast<unsigned char*>(address)-getAddress())/blockSize+1;
        getPool().returnSpace(off, req);
    }

    void constantSet(void* dest, const void * src, const unsigned int size, const cudaStream_t stream) {
        checkError(cudaMemcpyToSymbolAsync(memory,src,size,
            static_cast<unsigned char*>(dest)-getAddress(),cudaMemcpyHostToDevice,stream));
    }
}

