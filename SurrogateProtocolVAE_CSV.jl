module SurrogateProtocolVAE_CSV

using SurrogateProtocolVAE
import FileIO, CSV

function SurrogateProtocolVAE.load_entanglement_consumer_log(save_file::FileIO.File{FileIO.DataFormat{:CSV},String})
    df = CSV.read(save_file.filename, DataFrame)
    consumer_log = EntanglementConsumerLog()
    consumer_log.simulation_log.time = df.t
    consumer_log.simulation_log.state = hcat(df.obs1,df.obs2)

    return consumer_log
end

end