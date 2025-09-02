module Chatbot

export llm_chat

using Pkg

Pkg.activate("./JuliaBasics")

using LibPQ, UUIDs, DotEnv, HTTP, JSON3, Dates

DotEnv.load!("./env.env")

connection_string = "host=localhost port=5432 dbname=athena user=postgres password=Eushmetha1\$ sslmode=prefer"

function store_message(message::String, session_id::String, transaction_id=nothing)
    try
        if transaction_id === nothing
            transaction_id = string(UUIDs.uuid4())
            LibPQ.Connection(connection_string) do conn
                execute(conn, "INSERT INTO conversation_history (user_message, session_id, transaction_id) VALUES (\$1, \$2, \$3)", (message, session_id, transaction_id))
            end
            return transaction_id
        else
            LibPQ.Connection(connection_string) do conn
                execute(conn, "UPDATE conversation_history SET llm_response = \$1, llm_response_utc = \$2 WHERE transaction_id = \$3", (message, now(Dates.UTC), transaction_id))
            end
            return true
        end
    catch err
        rethrow(err)
        return false
    end
end

function get_conversation_history(session_id::String, n::Integer=5)
    try
        LibPQ.Connection(connection_string) do conn
            result = execute(conn, "SELECT user_message, llm_response FROM conversation_history WHERE session_id = \$1 ORDER BY id DESC LIMIT \$2", (session_id, n))
            return [collect(row) for row in reverse(collect(result))]
        end
    catch err
        rethrow(err)
        return []
    end
end

function llm_chat(message::String, session_id::String)
    transaction_id = store_message(message, session_id)
    conversation_history = []
    for (user_message, llm_response) in get_conversation_history(session_id)
        if !(ismissing(user_message) || isnothing(user_message))
            push!(conversation_history, Dict(["role" => "user", "content" => user_message]))
        end
        if !(ismissing(llm_response) || isnothing(llm_response))
            push!(conversation_history, Dict(["role" => "assistant", "content" => llm_response]))
        end
    end
    messages = [Dict(["role" => "system", "content" => "You are a helpful assistant."]); conversation_history]
    api_key = ENV["togetherai_api_key"]
    headers = Dict(["Content-Type" => "application/json", "Authorization" => "Bearer $(api_key)"])
    payload = Dict(["messages" => messages, "model" => "openai/gpt-oss-120b"])
    resp = HTTP.post(ENV["togetherai_api_endpoint"], body=JSON3.write(payload), headers=headers)
    resp_dict = JSON3.parse(String(copy(resp.body)))
    response = resp_dict["choices"][1]["message"]["content"]
    store_message(response, session_id, transaction_id)
    return response
end

end
