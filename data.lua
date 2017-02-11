--[[
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the CC-NC style license found in the
LICENSE file in the root directory of this source tree.
--]]

local datautils = {}

local function fetchdata(dset)
    local wwwlink
    local rootpath = '/tmp/rnnlm_datasets/text/'
    if dset == 'ptbw' then
        wwwlink = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
        local destpath = rootpath .. 'ptb/words/'
        os.execute('mkdir -p ' .. destpath)
        os.execute('wget ' .. wwwlink)
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.train.txt')
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.test.txt')
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.valid.txt')
        os.execute('mv simple-examples/data/ptb.train.txt ' .. destpath)
        os.execute('mv simple-examples/data/ptb.test.txt '  .. destpath)
        os.execute('mv simple-examples/data/ptb.valid.txt ' .. destpath)
        os.execute('rm -r simple-examples*')
    elseif dset == 'ptbc' then
        wwwlink = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
        local destpath = rootpath .. 'ptb/chars/'
        os.execute('mkdir -p ' .. destpath)
        os.execute('wget ' .. wwwlink)
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.char.train.txt')
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.char.test.txt')
        os.execute('tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.char.valid.txt')
        os.execute('mv simple-examples/data/ptb.char.train.txt ' .. destpath)
        os.execute('mv simple-examples/data/ptb.char.test.txt '  .. destpath)
        os.execute('mv simple-examples/data/ptb.char.valid.txt ' .. destpath)
        os.execute('rm -r simple-examples*')
    elseif dset == 'wikitext2' then
        wwwlink = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        local destpath = rootpath .. 'wikitext/wikitext-2/'
        os.execute('mkdir -p ' .. destpath)
        os.execute('wget ' .. wwwlink)
        os.execute('unzip wikitext-2-v1.zip')
        os.execute('mv wikitext-2/* ' .. destpath)
        os.execute('rm -r wikitext-2*')
    elseif dset == 'wikitext-103_large' then
        wwwlink = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
        local destpath = rootpath .. 'wikitext/wikitext-103/'
        os.execute('mkdir -p ' .. destpath)
        os.execute('wget ' .. wwwlink)
        os.execute('unzip wikitext-103-v1.zip')
        os.execute('mv wikitext-103/* ' .. destpath)
        os.execute('rm -r wikitext-103*')
    end
end

-- | Initialize a very basic dictionary object.
local function initdictionary()
    local dict = {idx2word = {},
                  word2idx = {},
                  idx2freq = {}
                 }
    return dict
end

local function addword(dict, word)
    if not dict.word2idx[word] then
        local id = #dict.idx2word + 1
        dict.word2idx[word] = id
        dict.idx2word[id]   = word
        dict.idx2freq[id]   = 1
    else
        local id = dict.word2idx[word]
        dict.idx2freq[id] = dict.idx2freq[id] + 1
    end
    return dict.word2idx[word]
end

-- | Tokenize a text file.
local function loadfile(path, dict)
    -- Read words from file.
    assert(paths.filep(path))
    local tokens = 0
    local sep = ' '
    for line in io.lines(path) do
        if line ~= ' ' then
            for word in string.gmatch(line, "([^" .. sep .. "]+)") do
                addword(dict, word)
                tokens = tokens + 1
                if tokens % 1000000 == 0 then
                    io.write(string.format('.')); io.flush()
                end
            end
            addword(dict, "<eos>")
            tokens = tokens + 1
        end
    end
    -- tensorize the data
    local ids = torch.LongTensor(tokens)
    local token = 1
    for line in io.lines(path) do
        if line ~= ' ' then
            for word in string.gmatch(line, "([^" .. sep .. "]+)") do
                ids[token] = dict.word2idx[word]
                token = token + 1
                if token % 1000000 == 0 then
                    io.write(string.format('.')); io.flush()
                end
            end
            ids[token] = dict.word2idx['<eos>']
            token = token + 1
        end
    end
    print('')
    -- Final dataset.
    return ids
end


datautils.setPaths = function (dset)
    local root_path = '/tmp/rnnlm_datasets/text/'
    local trpath, vlpath, tepath
    if dset == 'ptbw' then
        trpath = root_path .. 'ptb/words/ptb.train.txt'
        vlpath = root_path .. 'ptb/words/ptb.valid.txt'
        tepath = root_path .. 'ptb/words/ptb.test.txt'
    elseif dset == 'ptbc' then
        trpath = root_path .. 'ptb/chars/ptb.char.train.txt'
        vlpath = root_path .. 'ptb/chars/ptb.char.valid.txt'
        tepath = root_path .. 'ptb/chars/ptb.char.test.txt'
    elseif dset == 'wikitext-103_large' then
        trpath = root_path .. 'wikitext/wikitext-103/wiki.train.tokens'
        vlpath = root_path .. 'wikitext/wikitext-103/wiki.valid.tokens'
        tepath = root_path .. 'wikitext/wikitext-103/wiki.test.tokens'
    elseif dset == 'wikitext2' then
        trpath = root_path .. 'wikitext/wikitext-2/wiki.train.tokens'
        vlpath = root_path .. 'wikitext/wikitext-2/wiki.valid.tokens'
        tepath = root_path .. 'wikitext/wikitext-2/wiki.test.tokens'
    end
    return trpath, vlpath, tepath
end


datautils.loadData = function(opt, datapath)
    local dict, train, valid, test
    local dictbin = paths.concat(datapath, 'dict.th7')
    if not paths.filep(dictbin) then
        print('-- creating the data from scratch')
        if not paths.filep(opt.train) then fetchdata(opt.dset) end
        dict  = initdictionary()
        train = loadfile(opt.train, dict)
        valid = loadfile(opt.valid, dict)
        test  = loadfile(opt.test,  dict)
        -- save the files
        torch.save(paths.concat(datapath, 'dict.th7'),  dict)
        torch.save(paths.concat(datapath, 'train.th7'), train)
        torch.save(paths.concat(datapath, 'valid.th7'), valid)
        torch.save(paths.concat(datapath, 'test.th7'),  test)
    else
        print('-- loading the pre-tokenized data')
        dict  = torch.load(paths.concat(datapath, 'dict.th7'))
        train = torch.load(paths.concat(datapath, 'train.th7'))
        valid = torch.load(paths.concat(datapath, 'valid.th7'))
        test  = torch.load(paths.concat(datapath, 'test.th7'))
    end
    return {train = train,
            valid = valid,
            test = test,
            dict = dict}
end

return datautils
