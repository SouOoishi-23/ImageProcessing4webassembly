# Image processing for WebAssembly

## 環境
WSL:Ubuntu 20.04.3 LTS (GNU/Linux 4.4.0-19041-Microsoft x86_64)



## インストール
https://emscripten.org/docs/getting_started/downloads.html
```
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

一応毎回面倒なので.bash_profileとかに入れたほうがよさげ
```
echo 'source "/home/souoishi/emsdk/emsdk_env.sh"' >> $HOME/.bash_profile
```

開発版使うとめっちゃ更新かかるので定期的にpullする  
もしくは嫌なら`latest`->バージョンに書き換える
```
git pull
./emsdk install latest
./emsdk activate latest
```
## コンパイル
c++ から js等生成
```
emmake make
## em++ main.cpp -o main.js
```

## 表示
立ち上げ可能なブラウザ検索
```
$ emrun --list_browsers
emrun has automatically found the following browsers in the default install locations on the system:

  - firefox: Mozilla Firefox 92.0
  - chrome: Google Chrome

You can pass the --browser <id> option to launch with the given browser above.
Even if your browser was not detected, you can use --browser /path/to/browser/executable to launch with that browser.
```

```
emrun --browser firefox main.html
```

## コーディング
たぶんCしか動かないので関数の前にいろいろつける
```c
EMSCRIPTEN_KEEPALIVE
extern "C" int add(int a, int b)
{
  return a+b;
}
```

参照渡しはできるっぽいのでふつうに

## Separable Gaussian Filter
https://fukushimalab.github.io/hpc_exercise/boxfilter.html#%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%81%AB%E3%82%88%E3%82%8B%E9%AB%98%E9%80%9F%E5%8C%96

## SIMD
https://github.com/WebAssembly/simd  
[SIMD と webassemblyのSIMDの対応](https://emscripten.org/docs/porting/simd.html)
