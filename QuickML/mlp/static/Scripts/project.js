function ajax_request(request_type, url, data) {
    return new Promise(function (resolve, reject) {
        if (url === "/nn_validate_image/") {
            $.ajax({
                type: request_type,
                url: url,
                data: data,
                dataType: 'json',
                processData: false,
                contentType: false,
                success: function (response) {
                    resolve(response);
                },
                error: function (response) {
                    hideloading();
                    on_error(response);
                    reject();
                }
            });
        }
        else if (url === "/cps/"){
             $.ajax({
                type: request_type,
                url: url,
                data: data,
                dataType: 'json',
                processData: false,
                contentType: false,
                success: function (response) {
                    resolve(response);
                },
                error: function (response) {
                    hideloading();
                    on_error(response);
                    reject();
                }
            });
        }
        else {
            $.ajax({
                type: request_type,
                url: url,
                data: data,
                dataType: 'json',
                success: function (response) {
                    resolve(response);
                },
                error: function (response) {
                    hideloading();
                    on_error(response);
                    reject();
                }
            });
        }
    });
}

function on_error(response) {
    let path = window.location.origin + window.location.pathname;
    document.getElementById("wrapper_id").innerHTML = null;
    $("#nav_options_id").addClass("hidediv");
    let alert_div = "<div id='error_div' style='width: 100%;'>";

    let error_message = "<div class='card'>" +
        "<div class='card-header' style='font-weight: bold'>" +
        "Error Message</div>" +
        "<div id='error_message' style='font-weight: bold'  class='card-body'>" +
        "</div></div>";

    let error_response = "<div class='card'>" +
        "<div class='card-header' style='font-weight: bold' " +
        "role='button' data-toggle='collapse'  href='#error_response' " +
        "aria-expanded='false' aria-controls='collapseValid'> Error Info</div> " +
        "<div id='error_response' style='max-height: 5%; overflow-y: scroll;' " +
        "class='card-body panel-collapse collapse in' role='tabpanel'> </div></div>";

    let go_back = "<div class='card'> " +
        "<div class='card-header' style='font-weight: bold'> Want to go back??</div> " +
        "<div id='error_return' class='card-body'></div></div>";

    alert_div += error_message + error_response + go_back;
    alert_div += "</div>";

    let link = "<a class='btn' id='go_back'> Go Back </a>";
    document.getElementById("wrapper_id").innerHTML = alert_div;
    document.getElementById("error_message").innerText = "Something went wrong. We are looking into it. Thank You for your patience.";
    document.getElementById("error_response").innerText = response['responseText'];
    document.getElementById("error_return").innerHTML = link;
    document.getElementById("go_back").href = path;
    $("#project").removeClass("active");
}


function left_side_bar() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/getSidebar/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#sidebar').html(response['sidebar_html']);
            let p_type = response['p_type'];
            if (p_type == "Image Processing") {
                $('#settings-menu').addClass('hidediv');
            }
            hideloading();
        }
    });
}


function right_side_bar() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/right_side_bar/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            if(response['project_type']=="Data Processing") {
                $('#rightsidebar').html(response['right_side_bar_html']);
                $("#saved_models_info").html(response['saved_models_info_html']);
                $("#saved_cluster_models").html(response['saved_cluster_models_info_html']);
                $("#saved_regression_models").html(response['saved_regression_models_info_html']);
                hideloading();
            }
            else {
                if(response['saved_model']===1) {
                    $('#rightsidebar').html(response['right_side_bar_html']);
                    $("#model_info").removeClass("hidediv");
                    hideloading();
                }
                else {
                    $('#rightsidebar').html(response['right_side_bar_html']);
                    hideloading();
                }
            }
        }
    });
}


$(document).on('click', '#createprojectsubmit', function (event) {
    showloading();
    event.preventDefault();
    let data = new FormData();
    let p_name = $("#enterprojectname").val();
    let file = $('#choosefile')[0].files[0];
    data.append('project_file', file);
    data.append('project_name', p_name);
    data.append('csrfmiddlewaretoken',$('input[name=csrfmiddlewaretoken]').val());
    ajax_request("POST", "/cps/", data).then(function (response)
    {
            if (response !== null || response !== undefined)
            {
                let path = window.location.origin + response["new_url"];
                window.location.href=path;
                hideloading();
            }
    });
});


function go_back_one_step(event) {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0 || return_result === undefined) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {
        let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
        ajax_request("POST", "/undo/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                hideloading();
                alert("Dataframe is updated with previous data");
                location.reload();
            }
        });
    }
}

function select_project(project_id) {
    if (project_id === "") {
        return 0
    } else {
        return 1
    }
}

$(document).ready(function () {
    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/dpm/", data).then(function (response) {
        if (response !== null || response !== undefined) {
            $('#dropdown-menu').html(response['drop_down_html']);
            let project_id = $("#selectprojectbtn").val();
            let return_result = select_project(project_id);
            hideloading();
            if (return_result === 0) {
                document.getElementById('selectprojectbtn').innerText = "Select Project";
            } else {
                left_side_bar()
                right_side_bar()
            }
        }
    });
});

$(document).ready(function () {
    $('#createproject').on('click', function () {
        $('#project').addClass('active');
    });
    $('#createprojectclose').on('click', function () {
        $('#project').removeClass('active');
    });
});


$(document).on('change', '#p_type', function (event) {
    if (this.value === "Data Processing") {
        document.getElementById("choosefile").accept = ".xlsx, .xls, .csv";
    } else if (this.value == "Image Processing") {
        //$('#settings-menu').addClass('hidediv');
        document.getElementById("choosefile").accept = ".zip";
    }
});


function showloading() {
    $('#div_loader').css("display", "block")
}

function hideloading() {
    $("#div_loader").css("display", "none");
}

$(document).on('click', '#alertclose', function () {
    $("#alert").removeClass("showdiv");
});


//Summary of the Dataframe
$(document).on('click', '#summary', function (event) {
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
        ajax_request("POST", "/summary/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['summary_html']);
                $('#numerical_summary').html(response['numerical_summary_html']);
                $('#categorical_summary').html(response['categorical_summary_html']);
                hideloading();
            }
        });
    }
});

//View All Data
$(document).on('click', '#viewAllData', function (event) {
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
        ajax_request("POST", "/viewAllData/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['dataframe_html']);
                hideloading();
            }
        });
    }
});

//Delete Projects
function DeleteCheck() {
    val = confirm("Are You Sure you want to delete the project?");

    if (val === true) {
        deleteFunction();
    }
}

function deleteFunction(event) {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let return_result = select_project(project_id);
    if (return_result === 0) {
        document.getElementById("alertdata").innerHTML = "<p>please select project</p>";
        hideloading();
        $("#alert").addClass("showdiv");
    } else {
        let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
        ajax_request("POST", "/Deletion/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                window.location = window.location.protocol + "//" + window.location.hostname + ":" + window.location.port + "/";
            }
        });
    }
}