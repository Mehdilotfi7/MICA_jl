# MICA revision figure theme
# Include this file at the top of every Julia figure-generation script.

using Plots

const MICA_BLUE    = RGB(0/255, 114/255, 178/255)
const MICA_ORANGE  = RGB(230/255, 159/255, 0/255)
const MICA_GREEN   = RGB(0/255, 158/255, 115/255)
const MICA_PURPLE  = RGB(204/255, 121/255, 167/255)
const MICA_SKY     = RGB(86/255, 180/255, 233/255)
const MICA_RED     = RGB(213/255, 94/255, 0/255)
const MICA_YELLOW  = RGB(240/255, 228/255, 66/255)
const MICA_BLACK   = RGB(0/255, 0/255, 0/255)

const MICA_COLORS = [
    MICA_BLUE, MICA_ORANGE, MICA_GREEN, MICA_PURPLE,
    MICA_SKY, MICA_RED, MICA_YELLOW, MICA_BLACK
]

# PLOS allows Arial, Times, or Symbol. Use Arial as the default sans-serif.
# If Arial is not installed locally, GR falls back to a monospace font, which
# is unacceptable.  Liberation Sans is metric-compatible with Arial and is
# usually available on Linux; use it as the active fallback here.  For the
# final submission-ready EPS, ensure Arial (or an Arial-compatible substitute)
# is embedded.
const MICA_FONT = Plots.font(isfile(joinpath(homedir(), ".fonts", "Arial.ttf")) ? "Arial" : "DejaVu Sans")

function apply_mica_theme!()
    default(
        fontfamily        = MICA_FONT.family,
        titlefont         = MICA_FONT,
        guidefont         = MICA_FONT,
        tickfont          = MICA_FONT,
        legendfont        = MICA_FONT,
        titlefontsize     = 14,
        guidefontsize     = 12,
        tickfontsize      = 10,
        legendfontsize    = 10,
        linewidth         = 1.5,
        markersize        = 5,
        markerstrokewidth = 0,
        framestyle        = :box,
        grid              = true,
        gridalpha         = 0.2,
        gridlinewidth     = 0.5,
        dpi               = 300,
        size              = (800, 500),
        margin            = 5Plots.mm,
        label             = "",
        palette           = :tab10,  # overridden by explicit color vectors
    )
end

apply_mica_theme!()

# Convenience: reset color cycle to MICA_COLORS explicitly when needed.
mica_color_cycle() = MICA_COLORS
