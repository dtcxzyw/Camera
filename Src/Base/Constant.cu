#include <Base/Constant.hpp>
#include <algorithm>

namespace Impl {
    __constant__ unsigned char memory[blockNum*blockSize];

    class SegTree final {
    private:
        static constexpr auto treeSize = (blockNum + 1) << 2;
        uint siz[treeSize], lsiz[treeSize], rsiz[treeSize];
        u8 flag[treeSize];
#define ls l,m,id<<1
#define rs m+1,r,id<<1|1
        void build(uint l, uint r, uint id) {
            siz[id] = lsiz[id] = rsiz[id] = r - l + 1;
            flag[id] = 2;
            if (l != r) {
                auto m = (l + r) >> 1;
                build(ls);
                build(rs);
            }
        }
        void update(uint l, uint r, uint id) {
            auto m = (l + r) >> 1;
            lsiz[id] = lsiz[id << 1];
            if (lsiz[id] == m - l + 1)lsiz[id] += lsiz[id << 1 | 1];
            rsiz[id] = rsiz[id << 1|1];
            if (rsiz[id] == r - m)rsiz[id] += rsiz[id << 1];
            siz[id] = std::max(std::max(siz[id << 1], siz[id << 1 | 1]), rsiz[id << 1] + lsiz[id<<1|1]);
        }

        void color(uint l, uint r, uint id, u8 mark) {
            flag[id] = mark;
            siz[id] = lsiz[id] = rsiz[id] = (mark ? r - l + 1U : 0U);
        }

        void push(uint l, uint r, uint id) {
            if (flag[id] != 2) {
                auto m = (l + r) >> 1;
                color(ls, flag[id]);
                color(rs, flag[id]);
                flag[id] = 2;
            }
        }

        void modify(uint l, uint r, uint id, uint nl, uint nr,u8 mark) {
            if (nl <= l && r <= nr) color(l, r, id, mark);
            else {
                push(l, r, id);
                auto m = (l + r) >> 1;
                if (nl <= m)modify(ls, nl, nr, mark);
                if (m < nr)modify(rs, nl, nr, mark);
                update(l, r, id);
            }
        }

        uint query(uint l, uint r, uint id,uint size) {
            if (l == r)return l;
            push(l, r, id);
            auto m = (l + r) >> 1;
            if (siz[id << 1] >= size)return query(ls, size);
            if (rsiz[id << 1] + lsiz[id << 1 | 1] >= size)return m - rsiz[id << 1] + 1;
            return query(rs, size);
        }
    public:
        SegTree(){
            build(1, blockNum, 1);
        }
        uint allocSpace(uint size) {
            if (siz[1] >= size) {
                auto l = query(1, blockNum, 1, size);
                modify(1,blockNum,1,l,l+size-1,0);
                return l;
            }
            return 0;
        }
        void returnSpace(uint off,uint size) {
            modify(1, blockNum, 1, off, off + size - 1, 1);
        }
    };

    SegTree& getPool() {
        static thread_local SegTree mPool;
        return mPool;
    }

    auto getAddress() {
        static thread_local void* address=nullptr;
        if(address==nullptr)
            checkError(cudaGetSymbolAddress(&address, memory));
        return static_cast<unsigned char*>(address);
    }

    void* constantAlloc(unsigned int size) {
        auto req = calcSize(size,blockSize);
        auto off = getPool().allocSpace(req);
        if (off == 0)return nullptr;
        return getAddress()+(off-1)*blockSize;
    }

    void constantFree(void* address,unsigned int size) {
        auto req = calcSize(size, blockSize);
        auto off = (static_cast<unsigned char*>(address)-getAddress())/blockSize+1;
        getPool().returnSpace(off, req);
    }

    void constantSet(void* dest, const void * src, unsigned int size, cudaStream_t stream) {
        checkError(cudaMemcpyToSymbolAsync(memory,src,size,
            static_cast<unsigned char*>(dest)-getAddress(),cudaMemcpyHostToDevice,stream));
    }
}

