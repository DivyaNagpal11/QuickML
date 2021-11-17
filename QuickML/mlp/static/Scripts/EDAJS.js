//Visualization
$(document).on('click', '#visualization', function () {
    event.preventDefault();
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/plotting/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['plotting_html']);
                $('#distributionplotid').html(response['choose_features_distribution_html']);
                $('#distributionplotdata').html(response['distribution_plot_html']);
                $('#boxplotid').html(response['choose_features_box_html']);
                $('#boxplotdata').html(response['box_plot_html']);
                $('#scatterplotid').html(response['choose_features_scatter_html']);
                $('#scatterplotdata').html(response['scatter_plot_html']);
                $('#correlationplotheading').html(response['correlation_plot_heading']);
                $('#correlationplotdata').html(response['correlation_plot_html']);
                hideloading();
            }
        });
    }
});

//Distribution Plot
function distribution_plot() {
    showloading();
    let col = $('#distributionchartcol').val();
    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val(),
        "col": col
    };

    ajax_request("POST", "/distributionchart/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#distributionplotid').html(response['choose_features_html']);
            $('#distributionplotdata').html(response['distribution_plot_html']);
            hideloading();
        }

    });
}

//Box Plot
function box_plot() {
    showloading();
    let col1 = $('#boxplotcol1').val();
    let col2 = $('#boxplotcol2').val();
    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val(),
        "col1": col1,
        "col2": col2
    };
    ajax_request("POST", "/boxplot/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#boxplotid').html(response['choose_features_html']);
            $('#boxplotdata').html(response['box_plot_html']);
            hideloading();
        }

    });
}

//Scattter Plot
function scatter_plot() {
    showloading();
    let col1 = $('#scatterplotcol1').val();
    let col2 = $('#scatterplotcol2').val();
    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val(),
        "col1": col1,
        "col2": col2
    };
    ajax_request("POST", "/scatterplot/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#scatterplotid').html(response['choose_features_html']);
            $('#scatterplotdata').html(response['scatter_plot_html']);
            hideloading();
        }
    });
}


//Visualization Ends

//Data Cleaning Start
//Missing Values
$(document).on('click', '#missingvalues', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
        ajax_request("POST", "/missingvalues/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['missing_value_html']);
                hideloading();
            }
        });
    }
});

$(document).on('change', '#missingmethod', function () {
    let method = $(this).val();
    if (method === "value") {
        $(this).closest('tr').find("input").addClass("showdiv");
    } else {
        $(this).closest('tr').find("input").removeClass("showdiv");
    }
});

$(document).on('click', '#missingvaluessubmit', function () {
    showloading();
    let rows = [];
    getData();
    if (rows.length === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select a feature</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": $("#selectprojectbtn").val(),
                    "tabledata": JSON.stringify(rows)};

        ajax_request("POST", "/missingvalueimputation/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['missing_value_html']);
                right_side_bar();
                hideloading();
            }
        });
    }

    function getData() {
        $('#missingtabledata tbody tr').each(function () {
            let feature = $(this).find('td:eq(0)').text();
            let type = $(this).find('td:eq(1)').text();
            let count = $(this).find('td:eq(2)').text();
            let method = $(this).find('td:eq(3) option:selected').val();
            let value = $(this).find('td:eq(4) input').val();
            rows.push({
                feature,
                type,
                count,
                method,
                value
            });
        })
    }
});

//Normalization
$(document).on('click', '#normalization', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id};

        ajax_request("POST", "/normalization/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['normalization_html']);
                $('#distributionid').html(response['distribution_plot_choose_features_html']);
                $('#distributiondata').html(response['distribution_plot_html']);
                hideloading();
            }
        });
    }
});

$(document).on('change', '#normalizationmethod', function () {
    let method = $(this).val();
    if (method === "Normalization in range") {
        $(this).closest('tr').find("input").addClass("showdiv");
    } else {
        $(this).closest('tr').find("input").removeClass("showdiv");
    }
});

$(document).on('click', '#normalizationsubmit', function () {
    showloading();
    $("#alert").removeClass("showdiv");
    let rows = [];
    getData();
    if ($("#alert").css('display') == 'none') {
        if (rows.length === 0) {
            document.getElementById("alertdata").innerHTML = "<p>please choose methods for atleast 1 feature</p>";
            hideloading();
            $("#alert").addClass("showdiv");
        } else {

            let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                        "project_id": $("#selectprojectbtn").val(),
                        "tabledata": JSON.stringify(rows)};

            ajax_request("POST", "/normalizationmethods/", data).then(function(response){
                if (response !== null || response !== undefined){
                    $('#contentdata').html(response['normalization_html']);
                    $('#distributionid').html(response['distribution_plot_choose_features_html']);
                    $('#distributiondata').html(response['distribution_plot_html']);
                    right_side_bar();
                    hideloading();
                }
            });
        }
    }

    function getData() {
        $('#normalizationtabledata tbody tr').each(function () {
            let feature = $(this).find('td:eq(0)').text();
            let method = $(this).find('td:eq(3) option:selected').val();
            if (method === '') {
                return;
            }
            let lower = $(this).find('td:eq(4) input').val();
            let higher = $(this).find('td:eq(5) input').val();
            if (method === "Normalization in range") {
                if (lower === "" || higher === "") {
                    document.getElementById("alertdata").innerHTML = "<p>please enter the values</p>";
                    $("#alert").addClass("showdiv");
                    hideloading();
                    return;
                }
                rows.push({
                    feature,
                    lower,
                    higher,
                    method
                });
            } else {
                rows.push({
                    feature,
                    method
                })
            }
        })
    }
});

function distribution_plot_normalization() {
    showloading();
    event.preventDefault();

    let data = {
        'csrfmiddlewaretoken': '{{ csrf_token }}',
        "project_id": $("#selectprojectbtn").val(),
        "feature": $('#distribution_plot_normalization').val()
    };

    ajax_request("POST", "/distribution_plot_normalization/", data).then(function (response) {
        if (response !== null || response !== undefined) {
           $('#distributiondata').html(response['distribution_plot_html']);
           hideloading();
        }
    });
}

//Outliers
$(document).on('click', '#outlierdetection', function () {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/outliers/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['outliers_html']);
                $('#boxplotid').html(response['box_plot_choose_features_html']);
                $('#boxplotdata').html(response['box_plot_html']);
                hideloading();
            }
        });
    }
});

$(document).on('click', '#outlierssubmit', function () {
    showloading();
    let selected_features = [];
    get_selected_outliers_features();
    if (selected_features.length === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select a feature</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "tabledata": JSON.stringify(selected_features)
        };

        ajax_request("POST", "/outliers_handling/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['outliers_html']);
                $('#boxplotid').html(response['box_plot_choose_features_html']);
                $('#boxplotdata').html(response['box_plot_html']);
                right_side_bar();
                hideloading();
            }
        });
    }

    function get_selected_outliers_features() {
        $('#outlierstabledata tbody tr').each(function () {
            if (($(this).find('td:eq(2) input')).is(":checked")) {
                selected_features.push($(this).find('td:eq(0)').text());
            }
        })
    }
});

function box_plot_outliers() {
    showloading();
    let data = {
        'csrfmiddlewaretoken': '{{ csrf_token }}',
        "project_id": $("#selectprojectbtn").val(),
        "feature": $('#boxplotoutlierfeature').val()
    };

    ajax_request("POST", "/boxplot_outliers/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#boxplotdata').html(response['box_plot_html']);
            hideloading();
        }
    });
}

//Delete Columns
$(document).on('click', '#deletecolumn', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {
        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/deletefeature/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['delete_feature_html']);
                hideloading();
            }
        });
    }
});

$(document).on('click', '#delete_feature_submit', function () {
    showloading();
    let selected_features = [];
    get_selected_features();
    if (selected_features.length === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select a feature</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "tabledata": JSON.stringify(selected_features)
        };

        ajax_request("POST", "/deletefeatureshandling/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['delete_feature_html']);
                right_side_bar();
                hideloading();
            }
        });
    }

    function get_selected_features() {
        $('#delete_table_data tbody tr').each(function () {
            if (($(this).find('td:eq(1) input')).is(":checked")) {
                selected_features.push($(this).find('td:eq(0)').text());
            }
        })
    }
});
//Data Cleaning Ends

//Normal Distribution
$(document).on('click', '#distNormal', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/normalPlot/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['plotting_html']);
                $('#Normaldistributionplotid').html(response['choose_features_normal_distribution_html']);
                $('#Normaldistributionplotdata').html(response['normal_distribution_plot_html']);
                hideloading();
            }
        });
    }
});

function normal_distribution_plot() {
    showloading();
    let col = $('#normaldistributionchart').val();

    let data = {
        'csrfmiddlewaretoken': '{{ csrf_token }}',
        "project_id": $("#selectprojectbtn").val(),
        "col": col
    };

    ajax_request("POST", "/normalDistributionchart/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#Normaldistributionplotdata').html(response['normal_distribution_plot_html']);
            hideloading();
        }
    });
}

$(document).on('click', '#NormalDist', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    let col = $('#normaldistributionchart').val();

    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": project_id,
        "col": col
    };

    ajax_request("POST", "/normalDistributionFunc/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#contentdata').html(response['plotting_html']);
            $('#Normaldistributionplotid').html(response['choose_features_normal_distribution_html']);
            $('#Normaldistributionplotdata').html(response['normal_distribution_plot_html']);
            hideloading();
        }
    });
});

//Encoding
$(document).on('click', '#encoding', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/encoding/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['encoder_html']);
                hideloading();
            }
        });
    }
});

$(document).on('click', '#encodingmethodsubmit', function (event) {
    showloading();
    let selected_features = [];
    get_selected_features();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    if (selected_features.length === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select a feature</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id,
            "method": $("#encodingmethod").val(),
            "selected_features": JSON.stringify(selected_features)
        };

        ajax_request("POST", "/encoding_method/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['encoded_html']);
                right_side_bar();
                hideloading();
            }
        });
    }

    function get_selected_features() {
        $('#encoding_table_data tbody tr').each(function () {
            if (($(this).find('td:eq(1) input')).is(":checked")) {
                selected_features.push($(this).find('td:eq(0)').text());
            }
        })
    }
});

//Sampling
$(document).on('click', '#sampling', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/sampling/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['sampling_html']);
                hideloading();
            }
        });
    }
});

$(document).on('change', '#sampling_feature', function (event) {
    showloading();

    event.preventDefault();

    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": $("#selectprojectbtn").val(),
        "feature": $("#sampling_feature").val()
    };

    ajax_request("POST", "/sampling_info/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#sample_info').html(response['sample_info_html']);
            $('#sample_plot').html(response['sample_bar_plot']);
            hideloading();
        }
    });
});

$(document).on('click', '#sample_button_submit', function (event) {
    showloading();
    event.preventDefault();

    let feature = $("#sampling_feature").val()
    let method = $("#sampling_method").val()

    if(feature === null && method === null)
    {
        document.getElementById("alertdata").innerHTML = "<p>Please Select a Feature and a Method</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    }
    else if(feature === null){
        document.getElementById("alertdata").innerHTML = "<p>Please Select a Feature</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    }
    else if(method === null) {
        document.getElementById("alertdata").innerHTML = "<p>Please Select a Method</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    }
    else {
        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "feature": $("#sampling_feature").val(),
            "method": $("#sampling_method").val()
        };

        ajax_request("POST", "/sampling_methods/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#sample_info').html(response['sample_info_html']);
                $('#sample_plot').html(response['sample_bar_plot']);
                if (response['error_info'] == "1") //Response if Catagorical data is found on the dataset
                {
                    document.getElementById("alertdata").innerHTML = "<p>Error : Dataset has Catagorical Data. Smote cannot be implemented on Catagorical Data</p>";
                    hideloading();
                    $("#alert").addClass("showdiv");
                } else if (response['error_info'] == "2") //Response if values in a Class are not enough for SMOTE
                {
                    document.getElementById("alertdata").innerHTML = "<p>Error : Not enough values in class. Minimum 6 needed for Smote</p>";
                    hideloading();
                    $("#alert").addClass("showdiv");
                } else if (response['error_info'] == "3") //Response if Tomek cant be performed
                {
                    document.getElementById("alertdata").innerHTML = "<p>Error : Cannot Perform Tomek on this dataset</p>";
                    hideloading();
                    $("#alert").addClass("showdiv");
                } else if (response['error_info'] == "4") //Response if SMOTE cant be performed
                {
                    document.getElementById("alertdata").innerHTML = "<p>Error : Class Labels are of type float. Cannot perform SMOTE</p>";
                    hideloading();
                    $("#alert").addClass("showdiv");
                }
                right_side_bar();
                hideloading();
            }
        });
    }


});

//Binning Mechanism Code Below
$(document).on('click', '#binning', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": project_id
        };

        ajax_request("POST", "/binning/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                 $('#contentdata').html(response['binning_html']);
                 hideloading();
            }
        });
    }
});

//Binning Dropdown Method On Change
$(document).on('change', '#binningmethod', function () {
    let method = $(this).val();
    if (method === "") {
        $("#binOption").addClass("hidediv");
        $("#binName").addClass("hidediv");
        $("#binSelfLabels").addClass("hidediv");
        $(this).closest("tr").find("select.binOption").addClass("hidediv");
        $(this).closest("tr").find("input.binNum").addClass("hidediv");
        $(this).closest('tr').find("input.binLabels").addClass("hidediv");
    } else {
        $("#binOption").removeClass("hidediv");
        $(this).closest("tr").find("select.binOption").removeClass("hidediv");
    }
});


//Binning Dropdown Option On Change
$(document).on('change', '#binningoption', function () {
    let option = $(this).val();
    if (option === "Auto Labeling") {
        $("#binName").removeClass("hidediv");
        $(this).closest("tr").find("input.binNum").removeClass("hidediv");

        $("#binSelfLabel").addClass("hidediv");
        $(this).closest('tr').find("input.binLabels").addClass("hidediv");
    } else if (option === "Manual Labeling") {
        $("#binName").removeClass("hidediv");
        $("#binSelfLabel").removeClass("hidediv");
        $(this).closest("tr").find("input.binNum").addClass("hidediv");
        $(this).closest('tr').find("input.binLabels").removeClass("hidediv");
    } else {
        $("#binName").addClass("hidediv");
        $("#binSelfLabel").addClass("hidediv");
        $(this).closest("tr").find("input.binNum").addClass("hidediv");
        $(this).closest('tr').find("input.binLabels").addClass("hidediv");
    }
});

//Binning Submit button
$(document).on('click', '#binningsubmit', function () {
    showloading();
    let selected_features = [];
    selected_features = get_selected_binning_features(); //requests for the features selected by the user
    if (selected_features.length !== 0 && selected_features !== undefined) {

        let data = {
            'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
            "project_id": $("#selectprojectbtn").val(),
            "tabledata": JSON.stringify(selected_features)
        };

        ajax_request("POST", "/binninghandler/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['binning_html']);
                right_side_bar();
                hideloading();
            }
        });
    }

    function get_selected_binning_features()    //returns the selected features with respective methods as a array of dictionary
    {
        binnedData = [];
        let selected = false;
        let auto_check = false;
        let manual_check = false;
        $('#binningtabledata tbody tr').each(function () {
            tableData = {};
            tableData['feature'] = $(this).find('td:eq(0) ').text();
            let method = $(this).find('td:eq(1) option:selected').val();
            let option = $(this).find('td:eq(2) option:selected').val();

            if (method === '') //returns from the function inside the loop when the feature is not selected and moves to the next feature
            {
                return;
            } else if (option === '') // notifies the user when the user has not selected the labeling option
            {
                document.getElementById("alertdata").innerHTML = "<p>Kindly Select Labeling Option </p>";
                hideloading();
                $("#alert").addClass("showdiv");
                return 0;
            } else {
                selected = true;
                tableData['action'] = method;
                tableData['option'] = option;
                if (tableData['option'] === 'Manual Labeling') {
                    tableData['label'] = $(this).find('td:eq(4) input').val();

                    let labelCheck = tableData['label'].split(",");
                    let flag = false;
                    let x;
                    let i;
                    for (i = 0; i < labelCheck.length; i++) {

                        x = labelCheck[i];
                        if (x == "") {
                            flag = true;
                            break;
                        }
                        let ws = x.split(" ");
                        if (ws.length > 1) {
                            flag = true;
                            break;
                        }
                    }

                    if (labelCheck.length === 0 || flag === true) //Notifies the user when labels are not given properly
                    {
                        manual_check = true;
                        return 0;
                    }
                } else if (tableData['option'] === 'Auto Labeling') {
                    tableData['binNumber'] = parseInt($(this).find('td:eq(3) input').val());

                    if (tableData['binNumber'] <= 1 || (!Number.isInteger(tableData['binNumber']))) //notfies the user to give correct bin number as input
                    {
                        auto_check = true
                        return 0;
                    }
                }
            }
            binnedData.push(tableData);
        })

        if (selected === false) //Notifies the user when no features are selected for binning
        {
            document.getElementById("alertdata").innerHTML = "<p>Please Choose Method for atleast 1 feature</p>";
            hideloading();
            $("#alert").addClass("showdiv");
            return 0;
        } else if (auto_check === true && manual_check === true) {
            let ErrorText = "<p>Kindly Enter the Number of Bins you need </p>";
            ErrorText += "<br><p>Kindly Enter Labels Correctly separated by ',' and give single word labels</p>";
            document.getElementById("alertdata").innerHTML = ErrorText;
            hideloading();
            $("#alert").addClass("showdiv");
        } else if (auto_check === true) {
            document.getElementById("alertdata").innerHTML = "<p>Kindly Enter the number of Bins you need</p>";
            hideloading();
            $("#alert").addClass("showdiv");
        } else if (manual_check === true) {
            document.getElementById("alertdata").innerHTML = "<p>Kindly Enter Labels Correctly separated by ',' and give single word labels</p>";
            hideloading();
            $("#alert").addClass("showdiv");
        } else
            return binnedData;
    }
});