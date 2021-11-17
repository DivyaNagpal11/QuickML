$(document).on('click', '#clustering', function(event) {
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

        ajax_request("POST", "/clustering/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['clustering_html']);
                $('#cluster_models_select').html(response['models_list_html']);
                $("#cluster_models_add_process").addClass("showdiv");
                let models_data = response["saved_cluster_models"];
                if (models_data !== 0) {
                    $('#cluster_algorithm_table_data').addClass('showdiv');
                    models_data.forEach(function(each) {
                        let newserialno = document.getElementById("cluster_algorithm_table_body").rows.length;
                        if (newserialno === undefined) {
                            newserialno = 0;
                        }
                        let table = document.getElementById("cluster_algorithm_table_body");
                        let row = table.insertRow(newserialno);
                        let col1 = row.insertCell(0);
                        let col2 = row.insertCell(1);
                        let col3 = row.insertCell(2);
                        let col4 = row.insertCell(3);
                        let col5 = row.insertCell(4);
                        let col6 = row.insertCell(5);
                        let col7 = row.insertCell(6);
                        let col8 = row.insertCell(7);
                        let col9 = row.insertCell(8);
                        col1.innerText = newserialno + 1;
                        col2.innerText = each['algorithm'];
                        col3.innerText = each['k_cluster']
                        col4.innerHTML = '<button class="btn" id="edit_cluster_input_parameters">Edit</button>';
                        col5.innerHTML = '<button class="btn" id="run_cluster_algorithm">Run</button>';
                        col6.innerHTML = '<button class="btn hidediv" id="cluster_labeling">Labeling</button>';
                        col7.innerHTML = '<input id="clusLabel" class="hidediv" placeholder="Enter Labels" style="width:120px" required>' + '&nbsp&nbsp <button  type="button" class="btn hidediv" class="btn" id="save_labels">Save</button>';
                        $('#clusLabel').val(each['cluster_labels']);
                        col8.innerHTML = '<button  type="button" class="btn hidediv" class="btn" id="publish_cluster_model">Publish</button>';
                        if (each['api_url'] !== 0) {
                            col9.innerHTML = '<input type="text" value="' + each['api_url'] + '">'
                        }
                        $('#selecteclusterdalgorithm option[value="' + each["algorithm"] + '"]').remove();
                        $('#selecteclusterdalgorithm').prop('selectedIndex', 0);
                    });
                }
                hideloading();
            }
        });
    }
});

function get_selected_cluster_algorithm_parameter() {
    showloading();
    let selected_algorithm = $("#selecteclusterdalgorithm").val();

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": $("#selectprojectbtn").val(),
                "selected_algorithm": selected_algorithm};

    ajax_request("POST", "/get_cluster_algorithm_parameters/", data).then(function(response){
        if (response !== null || response !== undefined){
            let data_frame = response['data_result'];
            if(data_frame === 0)
            {
                document.getElementById("alertdata").innerHTML = "<p>Data Frame has catgorical Data</p>";
                $("#alert").addClass("showdiv");
                hideloading();
            }
            else
            {
                if(response['elbow_plot_points'] != 0)
                {
                    $('#elbow_clusterplot').removeClass("hidediv");
                    $('#elbowclusterplot').html(response['elbow_plot_points']);
                }
                let add_html = generate_cluster_input_parametrs_html(response);
                document.getElementById('cluster_models_input_parameters').innerHTML = "";
                document.getElementById('cluster_models_input_parameters').innerHTML = add_html;
                $("#cluster_models_input_parameters").addClass("showdiv");
                hideloading();
            }
        }
    });
}

function generate_cluster_input_parametrs_html(response) {
    let obj = response['input_parameters'];
    let add_html = '';
    for (let i = 0; i < obj.length; i++) {
        let temp = obj[i];
        add_html += "<div style='width: 25%;display: inline-block'> <p style='display: inline-block;padding-left: 15px; width: 75%;'>" + temp[0] + ": </p>";
        if (temp[1] === "input") {
            add_html += '<input style="width: 25%" type="number" value="' + temp[3] + '"></div>';
        } else {
            add_html += "<select style='width: 25%'>";
            let opt = temp[3];
            for (let j = 0; j < opt.length; j++) {
                add_html += '<option value="' + opt[j] + '">' + opt[j] + "</option>";
            }
            add_html += "</select></div>"
        }
    }
    add_html += '<button class="btn hidediv" style="float:right;" id="save_cluster_input_parameters">Save</button>';
    return add_html
}

$(document).on('click', '#addclusteralgorithm', function() {
    showloading();
    let selected_algorithm = $("#selecteclusterdalgorithm").val();
    let newserialno = document.getElementById("cluster_algorithm_table_body").rows.length;
    $("#cluster_algorithm_table_data").addClass("showdiv");
    if (selected_algorithm == null) {
        document.getElementById("alertdata").innerHTML = "<p>please select a algorithm</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        $('#selecteclusterdalgorithm option[value="' + selected_algorithm + '"]').remove();
        $('#selecteclusterdalgorithm').prop('selectedIndex', 0);

        let table = document.getElementById("cluster_algorithm_table_body");
        let row = table.insertRow(newserialno);
        let col1 = row.insertCell(0);
        let col2 = row.insertCell(1);
        let col3 = row.insertCell(2);
        let col4 = row.insertCell(3);
        let col5 = row.insertCell(4);
        let col6 = row.insertCell(5);
        let col7 = row.insertCell(6);
        let col8 = row.insertCell(7);
        col1.innerText = newserialno + 1;
        col2.innerText = selected_algorithm;
        col4.innerHTML = '<button class="btn" id="edit_cluster_input_parameters">Edit</button>';
        col5.innerHTML = '<button class="btn" id="run_cluster_algorithm">Run</button>';
        col6.innerHTML = '<button class="btn hidediv" id="cluster_labeling">Labeling</button>';
        col7.innerHTML = '<input id="clusLabel" class="hidediv" placeholder="Enter Labels" style="width:120px" required>' + '&nbsp&nbsp <button  type="button" class="btn hidediv" class="btn" id="save_labels">Save</button>';
        col8.innerHTML = '<button  type="button" class="btn hidediv" class="btn" id="publish_cluster_model">Publish</button>';

        let input_parameters = [];
        $('#cluster_models_input_parameters input, #cluster_models_input_parameters select').each(
            function(index) {
                let input = $(this).val();
                input_parameters.push(input)
            }
        );

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": $("#selectprojectbtn").val(),
                    "algorithm": selected_algorithm,
                    "input_parameters": JSON.stringify(input_parameters)};

        ajax_request("POST", "/save_cluster_algorithm_input_parameters/", data).then(function(response){
            if (response !== null || response !== undefined){
                hideloading();
            }
        });

        $("#cluster_models_input_parameters").removeClass("showdiv");
        $("#elbow_clusterplot").addClass("hidediv");
    }
});

$(document).on('click', '#run_cluster_algorithm', function(){
    showloading();
    if ($("#save_cluster_input_parameters").hasClass("showdiv")) {
        document.getElementById("alertdata").innerHTML = "<p>please save the edited input parameters</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {
        let data_options = [];
        let row_index = $(this).closest('tr').index();
        let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[row_index];
        let selected_algorithm = row.cells[1].innerText;
        selected_algorithm = $.trim(selected_algorithm);
        let algorithm_info = [];
        let labels = $("#clusLabel").val();
        let label;
        if(labels === "")
        {
            label = 'auto';
        }
        else
        {
            label = labels;
        }
        algorithm_info.push(row_index, selected_algorithm, label);

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": $("#selectprojectbtn").val(),
                    "algorithm_info": JSON.stringify(algorithm_info)};

        ajax_request("POST", "/cluster_algorithms/", data).then(function(data){
            if (data !== null || data !== undefined){
                let algorithms_data = data['data_result'];
                let rowid = algorithms_data[0];
                let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[rowid];
                row.cells[2].innerText = algorithms_data[2];
                $("#clusterplot").removeClass("hidediv");
                $('#clusterplot').html(data['plot_points']);
                $("#cluster_labeling").removeClass('hidediv')
                $("#publish_cluster_model").removeClass("hidediv");
                right_side_bar();
                hideloading();
            }
        });
    }
});

$(document).on('click', '#edit_cluster_input_parameters', function() {
    showloading();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[row_index];
    let selected_algorithm = row.cells[1].innerText;

    selected_algorithm = $.trim(selected_algorithm);

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "selected_algorithm": selected_algorithm,
                "project_id": $('#selectprojectbtn').val()};

    ajax_request("POST", "/edit_cluster_algorithm_parameters/", data).then(function(response){
        if (response !== null || response !== undefined){
            let add_html = generate_cluster_input_parametrs_html(response);
            document.getElementById('cluster_models_input_parameters').innerHTML = "";
            document.getElementById('cluster_models_input_parameters').innerHTML = add_html;
            let x = document.getElementById("selecteclusterdalgorithm");
            let option = document.createElement("option");
            option.text = selected_algorithm;
            option.value = selected_algorithm;
            x.add(option, x[0]);
            $('#selecteclusterdalgorithm').prop('selectedIndex', 0);
            document.getElementById("selecteclusterdalgorithm").disabled = true;
            $("#cluster_models_input_parameters").addClass("showdiv");
            $("#save_cluster_input_parameters").addClass("showdiv");
            $("#cluster_labeling").addClass('hidediv');
            $('#clusLabel').addClass('hidediv');
            $('#save_labels').addClass('hidediv');
            $("#publish_cluster_model").addClass("hidediv");
            $("#addclusteralgorithm").addClass("hidediv");
            $("#clusterplot").addClass("hidediv");
            if(selected_algorithm === 'KMeans')
            {
                if(response['elbow_plot_points'] != 0)
                {
                    $('#elbow_clusterplot').removeClass("hidediv")
                    $('#elbowclusterplot').html(response['elbow_plot_points']);
                }
            }
            hideloading();
        }
    });
});

$(document).on('click', '#save_cluster_input_parameters', function() {
    showloading();
    document.getElementById("selecteclusterdalgorithm").disabled = false;
    $("#save_cluster_input_parameters").removeClass("showdiv");
    let input_parameters = [];
    $('#cluster_models_input_parameters input, #cluster_models_input_parameters select').each(
        function() {
            let input = $(this).val();
            input_parameters.push(input)
        }
    );
    let algorithm = $('#selecteclusterdalgorithm').val();
    $("#cluster_models_input_parameters").removeClass("showdiv");

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": $("#selectprojectbtn").val(),
                "algorithm": algorithm,
                "input_parameters": JSON.stringify(input_parameters)};

    ajax_request("POST", "/save_cluster_algorithm_input_parameters/", data).then(function(response){
        if (response !== null || response !== undefined){
            hideloading();
        }
    });

    let newK = Number(input_parameters[0]);
    let oldK;
    let algorithms_in_table = [];
    $('#cluster_algorithm_table_body tr').each(function() {
        algorithms_in_table.push($(this).find('td:eq(1)').text());
        oldK = Number($(this).find('td:eq(2)').text());
    });

    if (algorithms_in_table.includes(algorithm)) {
        $('#selecteclusterdalgorithm option[value="' + algorithm + '"]').remove();
        $('#selecteclusterdalgorithm').prop('selectedIndex', 0);
    }
    $("#addclusteralgorithm").removeClass("hidediv");
    $("#elbow_clusterplot").addClass("hidediv");

    if(oldK==0 || newK!=oldK)
        $("#clusLabel").val('');
});

$(document).on('click', '#cluster_labeling', function() {
    $('#clusLabel').removeClass('hidediv');
    $('#save_labels').removeClass('hidediv');
    $("#publish_cluster_model").addClass("hidediv");
});


$(document).on('click', '#save_labels', function(){
    showloading();
    let flag = false;
    if ($("#save_cluster_input_parameters").hasClass("showdiv")) {
        document.getElementById("alertdata").innerHTML = "<p>please save the edited input parameters</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else
    {
        let data_options = [];
        let row_index = $(this).closest('tr').index();
        let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[row_index];
        let selected_algorithm = row.cells[1].innerText;
        let cluster_k = Number( row.cells[2].innerText);
        selected_algorithm = $.trim(selected_algorithm);
        let algorithm_info = [];
        let labels =  $("#clusLabel").val();
        let label;
        if(labels === "")
        {
            label = 'auto';
        }
        else
        {
            lenCheck = labels.split(',');
            if(lenCheck.length !== cluster_k){
                flag = true;
            }
            else {
                label = labels;
            }
        }
        if(flag === true)
        {
            document.getElementById("alertdata").innerHTML = "<p>Cluster Labels and Number of Clusters do not match</p>";
            $("#alert").addClass("showdiv");
            hideloading();
        }
        else
        {
            algorithm_info.push(row_index, selected_algorithm, label);

            let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                        "project_id": $("#selectprojectbtn").val(),
                        "algorithm_info": JSON.stringify(algorithm_info)};

            ajax_request("POST", "/cluster_algorithms/", data).then(function(response){
                if (response !== null || response !== undefined){
                    let algorithms_data = response['data_result'];
                    let rowid = algorithms_data[0];
                    let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[rowid];
                    row.cells[2].innerText = algorithms_data[2];
                    $("#clusterplot").removeClass("hidediv");
                    $('#clusterplot').html(response['plot_points']);
                    $('#clusLabel').addClass('hidediv');
                    $('#save_labels').addClass('hidediv');
                    $("#publish_cluster_model").removeClass("hidediv");
                    right_side_bar();
                    hideloading();
                }
            });
        }
    }
});

$(document).on('click', '#cluster_validation', function(event) {
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

        ajax_request("POST", "/cluster_validation/", data).then(function(response){
            if (response !== null || response !== undefined){
                $('#contentdata').html(response['validation_html']);
                hideloading();
            }
        });
    }
});

function cluster_validate_get_info() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let selected_model = $("#validation_cluster_selected_algorithm").val();

    let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                "project_id": project_id,
                "algorithm": selected_model};

    ajax_request("POST", "/cluster_validation_data/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#validation_cluster_data').html(response['validation_data_html']);
            hideloading();
        }
    });
}

function cluster_predict_output() {
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let algorithm = $("#validation_cluster_selected_algorithm").val();
    let input_data = cluster_get_input_model_data();
    if(input_data == 0)
    {
        document.getElementById("alertdata").innerHTML = "<p>Please enter the Values</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    }else{
        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": project_id,
                    "algorithm": algorithm,
                    "input_data": JSON.stringify(input_data)};

        ajax_request("POST", "/cluster_prediction/", data).then(function(response){
            if (response !== null || response !== undefined){
                document.getElementById("cluster_prediction_result").innerText = response['predicted_result'];
                hideloading();
            }
        });
    }
}

function cluster_get_input_model_data() {
    let data = [];
    let flag = false;
    $('#cluster_validation_input_table tbody tr').each(function() {
        let input_feature = $(this).find('td:eq(0)').text();
        let input_value = $(this).find('td:eq(1) input').val();
        let value = checkValidate(input_value);
        if (value == 0){
            flag = true;
            return;
        } else{
            data.push([$.trim(input_feature), input_value]);
        }
    });
    if(flag){
        return 0;
    }
    else {
        return data
    }
}

$(document).on('click', '#publish_cluster_model', function() {
    showloading();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("cluster_algorithm_table_body").getElementsByTagName("tr")[row_index];
    colval = row.cells.length;
    let selected_algorithm = row.cells[1].innerText;
    selected_algorithm = $.trim(selected_algorithm);
    if (row.cells[2].innerHTML === "") {
        document.getElementById("alertdata").innerHTML = "<p>please run the algorithm first</p>";
        $("#alert").addClass("showdiv");
        hideloading();
    } else {

        let data = {'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
                    "project_id": $("#selectprojectbtn").val(),
                    "algorithm": selected_algorithm};

        ajax_request("POST", "/cluster_save_model_for_api/", data).then(function(response){
            if (response !== null || response !== undefined){
                let api_url = response["api_url"];
                if(colval == 9)
                {
                    row.deleteCell(8);
                }
                let col = row.insertCell(8);
                col.innerHTML = '<input type="text" value="' + api_url + '">';
                hideloading();
            }
        });
    }
});