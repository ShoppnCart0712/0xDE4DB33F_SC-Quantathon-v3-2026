import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Image displayed to the LEFT of the map
left_image_path = "dead.png"

# Background image displayed underneath the probability map
background_image_path = "map.png"

# Probability at or below this value is transparent
TRANSPARENCY_THRESHOLD = 0.2

# Probability color
HOT_COLOR = (1.0, 0.0, 0.0)


# ============================================================
# BACKEND MODEL FUNCTION
# ============================================================

def query_model(latitude, longitude, time):
    """
    Placeholder for the backend model.

    Replace this function with the real model call.
    """

    probability = 0.0
    ece = 0.0

    return probability, ece


# ============================================================
# LOAD CSV DATA
# ============================================================

df = pd.read_csv(
    "submission.csv"
)


# ============================================================
# PARSE DATES
# ============================================================

df["issue_date"] = pd.to_datetime(
    df["issue_date"]
)


# ============================================================
# NORMALIZE COORDINATES
#
# Rounding prevents tiny floating-point differences from
# creating different cell IDs.
# ============================================================

df["cell_lat"] = (
    df["cell_lat"]
    .astype(float)
    .round(6)
)

df["cell_lon"] = (
    df["cell_lon"]
    .astype(float)
    .round(6)
)


# ============================================================
# CREATE CELL ID
#
# A tuple is safer than concatenating floats into strings.
# ============================================================

df["cell_id"] = list(
    zip(
        df["cell_lat"],
        df["cell_lon"]
    )
)


# ============================================================
# GET ALL UNIQUE CELLS
#
# IMPORTANT:
# This establishes ONE fixed ordering for all dates.
# ============================================================

cell_coordinates = (
    df[
        [
            "cell_id",
            "cell_lat",
            "cell_lon"
        ]
    ]
    .drop_duplicates(
        "cell_id"
    )
    .sort_values(
        [
            "cell_lat",
            "cell_lon"
        ]
    )
)


cell_ids = (
    cell_coordinates[
        "cell_id"
    ]
    .tolist()
)


latitudes = (
    cell_coordinates[
        "cell_lat"
    ]
    .to_numpy()
)


longitudes = (
    cell_coordinates[
        "cell_lon"
    ]
    .to_numpy()
)


# ============================================================
# CREATE DATE × CELL PROBABILITY TABLE
# ============================================================

probability_table = (
    df.pivot_table(
        index="issue_date",
        columns="cell_id",
        values="probability",
        aggfunc="mean"
    )
    .sort_index()
    .reindex(
        columns=cell_ids
    )
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

probabilities = (
    probability_table
    .to_numpy()
)


dates = (
    probability_table
    .index
    .to_numpy()
)


# ============================================================
# DATA DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("CSV DATA")
print("=" * 60)

print(
    "Number of dates:",
    len(dates)
)

print(
    "Number of cells:",
    len(cell_ids)
)

print(
    "Probability array shape:",
    probabilities.shape
)

print()


print("Probability range by date:")

for i in range(
    min(10, len(dates))
):

    row = probabilities[i]

    valid = row[
        ~np.isnan(row)
    ]

    if len(valid) == 0:

        print(
            dates[i],
            "NO DATA"
        )

    else:

        print(
            pd.Timestamp(
                dates[i]
            ).strftime(
                "%Y-%m-%d"
            ),
            "min=",
            np.min(valid),
            "max=",
            np.max(valid),
            "mean=",
            np.mean(valid)
        )

print("=" * 60)
print()


# ============================================================
# GET UNIQUE LATITUDE/LONGITUDE VALUES
# ============================================================

unique_lats = np.sort(
    np.unique(
        latitudes
    )
)

unique_lons = np.sort(
    np.unique(
        longitudes
    )
)


if len(unique_lats) < 2:

    raise ValueError(
        "Need at least two unique latitude "
        "values to determine tile size."
    )


if len(unique_lons) < 2:

    raise ValueError(
        "Need at least two unique longitude "
        "values to determine tile size."
    )


# ============================================================
# DETERMINE TILE SPACING
# ============================================================

lat_spacing = np.median(
    np.diff(
        unique_lats
    )
)

lon_spacing = np.median(
    np.diff(
        unique_lons
    )
)


# ============================================================
# CREATE LATITUDE EDGES
# ============================================================

lat_edges = np.concatenate(
    [
        [
            unique_lats[0]
            - lat_spacing / 2
        ],

        (
            unique_lats[:-1]
            + unique_lats[1:]
        ) / 2,

        [
            unique_lats[-1]
            + lat_spacing / 2
        ]
    ]
)


# ============================================================
# CREATE LONGITUDE EDGES
# ============================================================

lon_edges = np.concatenate(
    [
        [
            unique_lons[0]
            - lon_spacing / 2
        ],

        (
            unique_lons[:-1]
            + unique_lons[1:]
        ) / 2,

        [
            unique_lons[-1]
            + lon_spacing / 2
        ]
    ]
)


# ============================================================
# CREATE LATITUDE/LONGITUDE INDEX
# ============================================================

lat_index = {
    lat: i
    for i, lat in enumerate(
        unique_lats
    )
}


lon_index = {
    lon: i
    for i, lon in enumerate(
        unique_lons
    )
}


# ============================================================
# CONVERT 1D PROBABILITY VECTOR TO 2D MAP GRID
# ============================================================

def make_probability_grid(
    probability_values
):

    grid = np.full(
        (
            len(unique_lats),
            len(unique_lons)
        ),
        np.nan
    )


    for lat, lon, probability in zip(
        latitudes,
        longitudes,
        probability_values
    ):

        i = lat_index[
            lat
        ]

        j = lon_index[
            lon
        ]

        grid[i, j] = probability


    return grid


# ============================================================
# INITIAL PROBABILITY GRID
# ============================================================

prob_grid = make_probability_grid(
    probabilities[0]
)


# ============================================================
# TRANSPARENT -> RED COLORMAP
# ============================================================

colors = np.zeros(
    (256, 4)
)


for i in range(256):

    colors[i, 0] = HOT_COLOR[0]
    colors[i, 1] = HOT_COLOR[1]
    colors[i, 2] = HOT_COLOR[2]

    # Alpha increases with probability
    colors[i, 3] = (
        i / 255.0
    )


hot_zone_cmap = (
    LinearSegmentedColormap.from_list(
        "transparent_red",
        colors
    )
)


# ============================================================
# MAKE NaN VALUES TRANSPARENT
# ============================================================

hot_zone_cmap = (
    hot_zone_cmap.copy()
)

hot_zone_cmap.set_bad(
    (0, 0, 0, 0)
)


# ============================================================
# NORMALIZE PROBABILITIES
# ============================================================

vmin = (
    TRANSPARENCY_THRESHOLD
)


vmax = np.nanmax(
    probabilities
)


if vmax <= vmin:

    raise ValueError(
        "Maximum probability must be greater than "
        "the transparency threshold."
    )


norm = Normalize(
    vmin=vmin,
    vmax=vmax
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(15, 7),
    constrained_layout=True
)


# ============================================================
# FIGURE GRID
#
# LEFT IMAGE | MAP | QUERY PANEL
#
# Bottom row = date slider
# ============================================================

gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    width_ratios=[
        2.0,
        3.2,
        1.25
    ],
    height_ratios=[
        1.0,
        0.10
    ],
    hspace=0.05,
    wspace=0.08
)


# ============================================================
# LEFT IMAGE AXES
# ============================================================

image_ax = fig.add_subplot(
    gs[0, 0]
)

image_ax.axis(
    "off"
)


# ============================================================
# MAP AXES
# ============================================================

ax = fig.add_subplot(
    gs[0, 1]
)


# ============================================================
# QUERY PANEL
# ============================================================

panel_ax = fig.add_subplot(
    gs[0, 2]
)

panel_ax.axis(
    "off"
)


# ============================================================
# LEFT IMAGE
# ============================================================

try:

    left_image = plt.imread(
        left_image_path
    )

    image_ax.imshow(
        left_image,
        aspect="equal"
    )

    image_ax.axis(
        "off"
    )

    image_ax.set_aspect(
        "equal",
        adjustable="box"
    )

except FileNotFoundError:

    image_ax.text(
        0.5,
        0.5,
        (
            f"Image not found:\n"
            f"{left_image_path}"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="gray"
    )


# ============================================================
# BACKGROUND MAP IMAGE
# ============================================================

if background_image_path is not None:

    background_image = plt.imread(
        background_image_path
    )

    ax.imshow(
        background_image,

        extent=[
            lon_edges[0],
            lon_edges[-1],
            lat_edges[0],
            lat_edges[-1]
        ],

        aspect="auto",

        zorder=0
    )


# ============================================================
# DRAW PROBABILITY MAP
# ============================================================

mesh = ax.pcolormesh(
    lon_edges,
    lat_edges,
    prob_grid,

    cmap=hot_zone_cmap,

    norm=norm,

    shading="flat",

    zorder=1
)


# ============================================================
# IMPORTANT:
#
# We DO NOT use set_facecolors().
#
# The colormap now controls both color and transparency.
# ============================================================


# ============================================================
# COLORBAR
# ============================================================

colorbar = fig.colorbar(
    mesh,
    ax=ax,
    label="Earthquake probability",
    pad=0.02
)


colorbar.set_ticks(
    np.linspace(
        TRANSPARENCY_THRESHOLD,
        vmax,
        5
    )
)


# ============================================================
# MAP AXES
# ============================================================

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)

ax.set_aspect(
    "equal",
    adjustable="box"
)


# ============================================================
# DATE FORMATTER
# ============================================================

def format_date(
    date
):

    return pd.Timestamp(
        date
    ).strftime(
        "%Y-%m-%d"
    )


# ============================================================
# INITIAL DATE
# ============================================================

initial_date = format_date(
    dates[0]
)


# ============================================================
# MAP TITLE
# ============================================================

ax.set_title(
    (
        "Earthquake probability — "
        f"{initial_date}"
    )
)


# ============================================================
# DATE LABEL
# ============================================================

date_label = fig.text(
    0.50,
    0.025,
    initial_date,
    ha="center",
    fontsize=10
)


# ============================================================
# DATE SLIDER
# ============================================================

slider_ax = fig.add_subplot(
    gs[1, 1]
)


date_slider = Slider(
    ax=slider_ax,

    label="Date",

    valmin=0,

    valmax=len(dates) - 1,

    valinit=0,

    valstep=1,

    color="steelblue"
)


# ============================================================
# PANEL GEOMETRY
# ============================================================

def get_panel_geometry():

    panel_position = (
        panel_ax.get_position()
    )


    panel_left = (
        panel_position.x0
    )

    panel_bottom = (
        panel_position.y0
    )

    panel_width = (
        panel_position.width
    )

    panel_height = (
        panel_position.height
    )


    padding = (
        panel_width * 0.08
    )


    widget_left = (
        panel_left
        + padding
    )


    widget_width = (
        panel_width
        - 2 * padding
    )


    return (
        panel_left,
        panel_bottom,
        panel_width,
        panel_height,
        widget_left,
        widget_width
    )


(
    panel_left,
    panel_bottom,
    panel_width,
    panel_height,
    widget_left,
    widget_width
) = get_panel_geometry()


# ============================================================
# PANEL TITLE
# ============================================================

panel_title = fig.text(
    widget_left,

    panel_bottom
    + panel_height * 0.92,

    "Model Query",

    fontsize=12,

    fontweight="bold",

    ha="left"
)


# ============================================================
# LATITUDE INPUT
# ============================================================

lat_ax = fig.add_axes(
    [
        widget_left,

        panel_bottom
        + panel_height * 0.76,

        widget_width,

        panel_height * 0.065
    ]
)


lat_box = TextBox(
    lat_ax,
    "Latitude",
    initial=""
)


# ============================================================
# LONGITUDE INPUT
# ============================================================

lon_ax = fig.add_axes(
    [
        widget_left,

        panel_bottom
        + panel_height * 0.64,

        widget_width,

        panel_height * 0.065
    ]
)


lon_box = TextBox(
    lon_ax,
    "Longitude",
    initial=""
)


# ============================================================
# TIME INPUT
# ============================================================

time_ax = fig.add_axes(
    [
        widget_left,

        panel_bottom
        + panel_height * 0.52,

        widget_width,

        panel_height * 0.065
    ]
)


time_box = TextBox(
    time_ax,
    "Time",
    initial=initial_date
)


# ============================================================
# QUERY BUTTON
# ============================================================

button_ax = fig.add_axes(
    [
        widget_left,

        panel_bottom
        + panel_height * 0.40,

        widget_width,

        panel_height * 0.075
    ]
)


query_button = Button(
    button_ax,
    "Query Model"
)


# ============================================================
# RESULTS TABLE
# ============================================================

result_table_ax = fig.add_axes(
    [
        widget_left,

        panel_bottom
        + panel_height * 0.10,

        widget_width,

        panel_height * 0.25
    ]
)


result_table_ax.axis(
    "off"
)


table_data = [
    [
        "Metric",
        "Value"
    ],

    [
        "Latitude",
        "—"
    ],

    [
        "Longitude",
        "—"
    ],

    [
        "Time",
        "—"
    ],

    [
        "Probability",
        "—"
    ],

    [
        "ECE",
        "—"
    ],
]


result_table = result_table_ax.table(
    cellText=table_data,
    loc="center",
    cellLoc="left",
    colWidths=[
        0.55,
        0.45
    ]
)


result_table.auto_set_font_size(
    False
)


result_table.set_fontsize(
    9
)


result_table.scale(
    1,
    1.5
)


# ============================================================
# TABLE HEADER
# ============================================================

def style_table_header():

    for column in range(2):

        cell = result_table[
            (0, column)
        ]

        cell.set_text_props(
            weight="bold"
        )

        cell.set_facecolor(
            "#e6e6e6"
        )


style_table_header()


# ============================================================
# UPDATE TABLE
# ============================================================

def update_table(
    values
):

    for row, row_values in enumerate(
        values
    ):

        for column, value in enumerate(
            row_values
        ):

            cell = result_table[
                (row, column)
            ]

            cell.get_text().set_text(
                str(value)
            )


    style_table_header()


# ============================================================
# SHOW TABLE ERROR
# ============================================================

def show_table_error(
    status,
    message
):

    values = [
        [
            "Status",
            status
        ],

        [
            "Message",
            message
        ],

        [
            "",
            ""
        ],

        [
            "",
            ""
        ],

        [
            "",
            ""
        ],

        [
            "",
            ""
        ],
    ]


    update_table(
        values
    )


# ============================================================
# RESIZE TABLE POSITION
# ============================================================

def update_table_position():

    (
        panel_left,
        panel_bottom,
        panel_width,
        panel_height,
        widget_left,
        widget_width
    ) = get_panel_geometry()


    result_table_ax.set_position(
        [
            widget_left,

            panel_bottom
            + panel_height * 0.10,

            widget_width,

            panel_height * 0.25
        ]
    )


# ============================================================
# DYNAMIC FONT SIZING
# ============================================================

def get_font_scale():

    width, height = (
        fig.get_size_inches()
    )


    scale = (
        width / 15.0
    )


    return np.clip(
        scale,
        0.75,
        1.50
    )


def update_fonts():

    scale = get_font_scale()


    ax.xaxis.label.set_size(
        10 * scale
    )

    ax.yaxis.label.set_size(
        10 * scale
    )

    ax.title.set_size(
        12 * scale
    )

    colorbar.ax.yaxis.label.set_size(
        9 * scale
    )

    date_label.set_fontsize(
        10 * scale
    )

    panel_title.set_fontsize(
        12 * scale
    )

    result_table.set_fontsize(
        9 * scale
    )


# ============================================================
# MODEL QUERY CALLBACK
# ============================================================

def query_model_from_form(
    event
):

    try:

        # ----------------------------------------------------
        # Read inputs
        # ----------------------------------------------------

        latitude = float(
            lat_box.text.strip()
        )


        longitude = float(
            lon_box.text.strip()
        )


        query_time = (
            time_box.text.strip()
        )


        # ----------------------------------------------------
        # Validate time
        # ----------------------------------------------------

        if not query_time:

            raise ValueError(
                "Time cannot be empty."
            )


        # ----------------------------------------------------
        # Validate latitude
        # ----------------------------------------------------

        if not -90 <= latitude <= 90:

            raise ValueError(
                (
                    "Latitude must be between "
                    "-90 and 90."
                )
            )


        # ----------------------------------------------------
        # Validate longitude
        # ----------------------------------------------------

        if not -180 <= longitude <= 180:

            raise ValueError(
                (
                    "Longitude must be between "
                    "-180 and 180."
                )
            )


        # ----------------------------------------------------
        # Call model
        # ----------------------------------------------------

        probability, ece = (
            query_model(
                latitude,
                longitude,
                query_time
            )
        )


        # ----------------------------------------------------
        # Update results table
        # ----------------------------------------------------

        values = [
            [
                "Metric",
                "Value"
            ],

            [
                "Latitude",
                f"{latitude:.4f}"
            ],

            [
                "Longitude",
                f"{longitude:.4f}"
            ],

            [
                "Time",
                query_time
            ],

            [
                "Probability",
                f"{probability:.4f}"
            ],

            [
                "ECE",
                f"{ece:.4f}"
            ],
        ]


        update_table(
            values
        )


    except ValueError as e:

        show_table_error(
            "Input error",
            str(e)
        )


    except Exception as e:

        show_table_error(
            "Model error",
            str(e)
        )


    fig.canvas.draw_idle()


# ============================================================
# CONNECT QUERY BUTTON
# ============================================================

query_button.on_clicked(
    query_model_from_form
)


# ============================================================
# SLIDER UPDATE
# ============================================================

def update(
    index
):

    # --------------------------------------------------------
    # Convert slider value to integer
    # --------------------------------------------------------

    index = int(
        index
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if index < 0:

        index = 0


    if index >= len(
        dates
    ):

        index = (
            len(dates) - 1
        )


    # --------------------------------------------------------
    # GET PROBABILITIES FOR THIS DATE
    #
    # This row contains the probability for every cell.
    # --------------------------------------------------------

    new_values = (
        probabilities[index]
    )


    # --------------------------------------------------------
    # CONVERT TO 2D MAP GRID
    # --------------------------------------------------------

    new_grid = (
        make_probability_grid(
            new_values
        )
    )


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    valid_values = (
        new_grid[
            ~np.isnan(new_grid)
        ]
    )


    if len(valid_values) > 0:

        print(
            "Slider index:",
            index,
            "| Date:",
            format_date(
                dates[index]
            ),
            "| min:",
            np.min(
                valid_values
            ),
            "| max:",
            np.max(
                valid_values
            ),
            "| mean:",
            np.mean(
                valid_values
            )
        )


    # --------------------------------------------------------
    # UPDATE MAP DATA
    #
    # IMPORTANT:
    #
    # Do NOT call set_facecolors().
    #
    # set_array() changes the actual data that the
    # ScalarMappable uses to calculate the colors.
    # --------------------------------------------------------

    mesh.set_array(
        new_grid.ravel()
    )


    # --------------------------------------------------------
    # UPDATE MAP TITLE
    # --------------------------------------------------------

    current_date = format_date(
        dates[index]
    )


    ax.set_title(
        (
            "Earthquake probability — "
            f"{current_date}"
        )
    )


    # --------------------------------------------------------
    # UPDATE DATE LABEL
    # --------------------------------------------------------

    date_label.set_text(
        current_date
    )


    # --------------------------------------------------------
    # UPDATE TIME TEXTBOX
    #
    # This means moving the slider also changes the
    # time shown in the query panel.
    # --------------------------------------------------------

    time_box.set_val(
        current_date
    )


    # --------------------------------------------------------
    # REDRAW
    # --------------------------------------------------------

    fig.canvas.draw_idle()


# ============================================================
# CONNECT SLIDER
# ============================================================

date_slider.on_changed(
    update
)


# ============================================================
# RESIZE HANDLING
# ============================================================

def on_resize(
    event
):

    # --------------------------------------------------------
    # Recalculate panel geometry
    # --------------------------------------------------------

    (
        panel_left,
        panel_bottom,
        panel_width,
        panel_height,
        widget_left,
        widget_width
    ) = get_panel_geometry()


    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    panel_title.set_position(
        (
            widget_left,

            panel_bottom
            + panel_height * 0.92
        )
    )


    # --------------------------------------------------------
    # Latitude textbox
    # --------------------------------------------------------

    lat_ax.set_position(
        [
            widget_left,

            panel_bottom
            + panel_height * 0.76,

            widget_width,

            panel_height * 0.065
        ]
    )


    # --------------------------------------------------------
    # Longitude textbox
    # --------------------------------------------------------

    lon_ax.set_position(
        [
            widget_left,

            panel_bottom
            + panel_height * 0.64,

            widget_width,

            panel_height * 0.065
        ]
    )


    # --------------------------------------------------------
    # Time textbox
    # --------------------------------------------------------

    time_ax.set_position(
        [
            widget_left,

            panel_bottom
            + panel_height * 0.52,

            widget_width,

            panel_height * 0.065
        ]
    )


    # --------------------------------------------------------
    # Query button
    # --------------------------------------------------------

    button_ax.set_position(
        [
            widget_left,

            panel_bottom
            + panel_height * 0.40,

            widget_width,

            panel_height * 0.075
        ]
    )


    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    update_table_position()


    # --------------------------------------------------------
    # Date label
    # --------------------------------------------------------

    map_position = (
        ax.get_position()
    )


    date_label.set_position(
        (
            map_position.x0
            + map_position.width / 2,

            0.025
        )
    )


    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------

    update_fonts()


    # --------------------------------------------------------
    # Redraw
    # --------------------------------------------------------

    fig.canvas.draw_idle()


# ============================================================
# CONNECT RESIZE EVENT
# ============================================================

fig.canvas.mpl_connect(
    "resize_event",
    on_resize
)


# ============================================================
# INITIAL FONT SIZING
# ============================================================

update_fonts()


# ============================================================
# INITIAL TABLE POSITION
# ============================================================

update_table_position()


# ============================================================
# DISPLAY
# ============================================================

plt.show()
