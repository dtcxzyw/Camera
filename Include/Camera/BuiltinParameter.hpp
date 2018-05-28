#pragma once

//in mm
namespace FocalLength {
    constexpr auto superWide = 15.0f;
    constexpr auto wideAngle = 25.0f;
    constexpr auto classic = 36.0f;
    constexpr auto normalLens = 50.0f;
    constexpr auto portrait = 80.0f;
    constexpr auto tele = 135.0f;
    constexpr auto superTele = 300.0f;
}

///@see <a href="https://en.wikipedia.org/wiki/List_of_motion_picture_film_formats">Common Film Formats</a>
namespace FileGate {
    constexpr float fullAperture[2] = {36.0f, 24.0f};
    constexpr float apsH[2] = {28.7f, 17.8f};
    constexpr float apsC[2] = {22.7f, 15.5f};
}
