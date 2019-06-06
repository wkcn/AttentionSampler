template <typename T>
MOBULA_KERNEL cumsum(T* X, T* I, const int N, const int att_size) {
    parfor(N, [&](int i) {
        T* Xi = X + i * att_size;
        T* Ii = I + i * att_size;
        for (int i = 1; i < att_size; ++i) {
            Ii[i] = Ii[i - 1] + Xi[i];
        }
    });
}

template <typename T>
MOBULA_KERNEL map_step(T* attxi, T* index_x, const int N, const float stepx, const float att_size, const int out_size) {
    T* myscale = 2.0 / (att_size - 1);
    parfor(N, [&](int b) {
        int i = 0;
        int j = 0;
        T* mapxi = attxi + b * att_size;
        T* index_i = index_x + b * att_size;
        while (i < out_size)
        {
            if (mapxi[j] >= i*stepx)
            {
                index_i[i] = (j + (i*stepx - mapxi[j]) / (mapxi[j] - mapxi[j - 1])) * myscale - 1.0;
                i++;
            }
            else {
                j++;
            }
        }
    });
}